#include <iostream>
#include <cmath>
#include <memory>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

void generateMatrix(float *matrix, int n, std::mt19937 &mt)
{
    std::uniform_real_distribution<float> dist(1.0f, 100.0f);
    for (int i = 0; i < n; ++i)
        matrix[i] = dist(mt);
}

__global__ void compute_attention_scores_qk(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    float *__restrict__ scores, // Output: [B, H, L, L]
    int B, int H, int L, int D)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z / H; // batch index
    int h = blockIdx.z % H; // head index
    if (q >= L || k >= L || b >= B)
        return;

    // Properly compute base indices considering separate b and h
    int base_q = ((b * H + h) * L + q) * D;
    int base_k = ((b * H + h) * L + k) * D;

    float score = 0.0f;
    for (int d = 0; d < D; ++d)
        score += Q[base_q + d] * K[base_k + d];

    // Properly index into scores using separate b and h
    scores[((b * H + h) * L + q) * L + k] = score / sqrtf((float)D);
}

__global__ void softmax_rows(
    float *__restrict__ scores, // in-place or out-of-place
    int B, int H, int L)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int q = threadIdx.x + blockIdx.z * blockDim.x;

    if (q >= L)
        return;

    int row_offset = ((b * H + h) * L + q) * L;
    float max_val = scores[row_offset];

    // 1. Find max for numerical stability
    for (int k = 1; k < L; ++k)
    {
        float val = scores[row_offset + k];
        max_val = fmaxf(max_val, val);
    }

    // 2. Compute exp and sum
    float sum = 0.0f;
    for (int k = 0; k < L; ++k)
    {
        float val = expf(scores[row_offset + k] - max_val);
        scores[row_offset + k] = val;
        sum += val;
    }

    // 3. Normalize
    for (int k = 0; k < L; ++k)
    {
        scores[row_offset + k] /= sum;
    }
}

__global__ void apply_value_weights(
    const float *__restrict__ softmax, // [B, H, L, L]
    const float *__restrict__ V,       // [B, H, L, D]
    float *__restrict__ output,        // [B, H, L, D]
    int B, int H, int L, int D)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int q = threadIdx.y + blockIdx.z * blockDim.y;

    if (q >= L)
        return;

    for (int d = threadIdx.x; d < D; d += blockDim.x)
    {
        float out = 0.0f;
        for (int k = 0; k < L; ++k)
        {
            int score_idx = ((b * H + h) * L + q) * L + k;
            int v = ((b * H + h) * L + k) * D + d;
            out += softmax[score_idx] * V[v];
        }

        int out_idx = ((b * H + h) * L + q) * D + d;
        output[out_idx] = out;
    }
}

__global__ void fused_multihead_attention(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ output,
    float *__restrict__ scores_buffer,
    int B, int H, int L, int D)
{
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int q_idx = threadIdx.x + blockIdx.z * blockDim.x;
    if (q_idx >= L)
        return;

    const float scale = 1.0f / sqrtf((float)D);
    int base_qkv = ((batch * H + head) * L + q_idx) * D;
    size_t scores_offset = ((size_t(batch) * H + head) * L + q_idx) * L; // space per thread
    float *scores = &scores_buffer[scores_offset];

    // 1. Compute dot products between Q[q_idx] and all K[k]
    for (int k = 0; k < L; ++k)
    {
        float score = 0.0f;
        for (int d = 0; d < D; ++d)
        {
            int qd = base_qkv + d;
            int kd = ((batch * H + head) * L + k) * D + d;
            score += Q[qd] * K[kd];
        }
        scores[k] = score * scale;
    }

    // 2. Softmax
    float max_score = scores[0];
    for (int i = 1; i < L; ++i)
        max_score = fmaxf(max_score, scores[i]);

    float sum = 0.0f;
    for (int i = 0; i < L; ++i)
    {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }

    for (int i = 0; i < L; ++i)
        scores[i] /= sum;

    // 3. Weighted sum over V
    for (int d = 0; d < D; ++d)
    {
        float out = 0.0f;
        for (int k = 0; k < L; ++k)
        {
            int vd = ((batch * H + head) * L + k) * D + d;
            out += scores[k] * V[vd];
        }
        output[base_qkv + d] = out;
    }
}

int main()
{
    const int B = 4, H = 8, L = 64, D = 32;
    const int size = B * H * L * D;
    const size_t bytes = size * sizeof(float);
    const int score_size = B * H * L * L;

    std::unique_ptr<float[]> h_Q(new float[size]);
    std::unique_ptr<float[]> h_K(new float[size]);
    std::unique_ptr<float[]> h_V(new float[size]);
    std::unique_ptr<float[]> h_O(new float[size]);
    std::unique_ptr<float[]> h_O_2(new float[size]);

    // std::mt19937 mt(std::random_device{}());
    std::mt19937 mt(42);
    generateMatrix(h_Q.get(), size, mt);
    generateMatrix(h_K.get(), size, mt);
    generateMatrix(h_V.get(), size, mt);

    float *d_Q, *d_K, *d_V, *d_O, *d_scores, *d_O_2, *d_scores_2;
    cudaMalloc((void **)&d_Q, bytes);
    cudaMalloc((void **)&d_K, bytes);
    cudaMalloc((void **)&d_V, bytes);
    cudaMalloc((void **)&d_O, bytes);
    cudaMalloc((void **)&d_O_2, bytes);
    cudaMalloc(&d_scores, score_size * sizeof(float));
    cudaMalloc(&d_scores_2, score_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, h_O.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_O_2, h_O.get(), bytes, cudaMemcpyHostToDevice);

    const int WARP_SIZE = 32;
    dim3 block_1(WARP_SIZE, 8); // 256 threads per block (optimal occupancy)
    dim3 grid_1((L + block_1.x - 1) / block_1.x, (L + block_1.y - 1) / block_1.y, B * H);
    compute_attention_scores_qk<<<grid_1, block_1>>>(d_Q, d_K, d_scores, B, H, L, D);

    int T_2 = 128;
    dim3 block_2(T_2);
    dim3 grid_2(B, H, (L + T_2 - 1) / T_2);
    softmax_rows<<<grid_2, block_2>>>(d_scores, B, H, L);

    dim3 block_3(WARP_SIZE, 8); // 256 threads max, align with warp size
    dim3 grid_3(B, H, (L + block_3.y - 1) / block_3.y);
    apply_value_weights<<<grid_3, block_3>>>(d_scores, d_V, d_O, B, H, L, D);

    cudaMemcpy(h_O.get(), d_O, bytes, cudaMemcpyDeviceToHost);

    int T = 128;
    dim3 block(T); // 128 threads per block
    dim3 grid(B, H, (L + block.x - 1) / block.x);

    global_multihead_attention<<<grid, block>>>(d_Q, d_K, d_V, d_O_2, d_scores_2, B, H, L, D);

    cudaMemcpy(h_O_2.get(), d_O_2, bytes, cudaMemcpyDeviceToHost);

    // Error check
    for (int i = 0; i < B * H * L * D; ++i)
        std::cout << (abs(h_O[i] - h_O_2[i]) < 0.0001);
    std::cout << std::endl;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_scores);
    cudaFree(d_O_2);
    cudaFree(d_scores_2);
    return 0;
}
