#include <iostream>
#include <cmath>
#include <memory>
#include <random>
#include <cuda_runtime.h>

void generateMatrix(float *matrix, int n, std::mt19937 &mt)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i)
        matrix[i] = dist(mt);
}

__global__ void fused_multihead_attention(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ output,
    float *__restrict__ scores_buffer,
    int B, int H, int L, int D)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int q = threadIdx.x + blockIdx.z * blockDim.x;
    if (q >= L)
        return;

    const float scale = 1.0f / sqrtf((float)D);
    int base_qkv = ((b * H + h) * L + q) * D;
    size_t scores_offset = ((size_t(b) * H + h) * L + q) * L; // space per thread
    float *scores = &scores_buffer[scores_offset];

    // 1. Compute dot products between Q[q] and all K[k]
    for (int k = 0; k < L; ++k)
    {
        float score = 0.0f;
        for (int d = 0; d < D; ++d)
        {
            int qd = base_qkv + d;
            int kd = ((b * H + h) * L + k) * D + d;
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
            int vd = ((b * H + h) * L + k) * D + d;
            out += scores[k] * V[vd];
        }
        output[base_qkv + d] = out;
    }
}

int main()
{
    const int B = 4, H = 8, L = 32, D = 16;
    const int size = B * H * L * D;
    const size_t bytes = size * sizeof(float);
    const int score_size = B * H * L * L;

    std::unique_ptr<float[]> h_Q(new float[size]);
    std::unique_ptr<float[]> h_K(new float[size]);
    std::unique_ptr<float[]> h_V(new float[size]);
    std::unique_ptr<float[]> h_O(new float[size]);

    // std::mt19937 mt(std::random_device{}());
    std::mt19937 mt(42);
    generateMatrix(h_Q.get(), size, mt);
    generateMatrix(h_K.get(), size, mt);
    generateMatrix(h_V.get(), size, mt);

    float *d_Q, *d_K, *d_V, *d_O, *d_scores;
    cudaMalloc((void **)&d_Q, bytes);
    cudaMalloc((void **)&d_K, bytes);
    cudaMalloc((void **)&d_V, bytes);
    cudaMalloc((void **)&d_O, bytes);
    cudaMalloc(&d_scores, score_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.get(), bytes, cudaMemcpyHostToDevice);

    int T = 128;
    dim3 block(T); // 128 threads per block
    dim3 grid(B, H, (L + block.x - 1) / block.x);

    fused_multihead_attention<<<grid, block>>>(d_Q, d_K, d_V, d_O, d_scores, B, H, L, D);

    cudaMemcpy(h_O.get(), d_O, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Sample output[0][0][0][:5]: ";
    for (int i = 0; i < 5; ++i)
        std::cout << h_O[i] << " ";
    std::cout << std::endl;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_scores);
    return 0;
}
