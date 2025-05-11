#include <iostream>
#include <cmath>
#include <memory>
#include <random>
#include <cuda_runtime.h>
#include <chrono>

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

int main(int argc, char **argv)
{
    // Default values
    int B = 4, H = 8, L = 32, D = 16;
    int iterations = 10;
    
    // Parse command line args
    if (argc > 1) B = atoi(argv[1]);
    if (argc > 2) H = atoi(argv[2]);
    if (argc > 3) L = atoi(argv[3]);
    if (argc > 4) D = atoi(argv[4]);
    if (argc > 5) iterations = atoi(argv[5]);
    
    const int size = B * H * L * D;
    const size_t bytes = size * sizeof(float);
    const int score_size = B * H * L * L;

    std::cout << "Running with: B=" << B << ", H=" << H << ", L=" << L << ", D=" << D 
              << ", iterations=" << iterations << std::endl;

    std::unique_ptr<float[]> h_Q(new float[size]);
    std::unique_ptr<float[]> h_K(new float[size]);
    std::unique_ptr<float[]> h_V(new float[size]);
    std::unique_ptr<float[]> h_O(new float[size]);

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

    // Warmup
    fused_multihead_attention<<<grid, block>>>(d_Q, d_K, d_V, d_O, d_scores, B, H, L, D);
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        fused_multihead_attention<<<grid, block>>>(d_Q, d_K, d_V, d_O, d_scores, B, H, L, D);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double total_ms = elapsed.count();
    double avg_ms = total_ms / iterations;
    
    cudaMemcpy(h_O.get(), d_O, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Sample output[0][0][0][:5]: ";
    for (int i = 0; i < 5; ++i)
        std::cout << h_O[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Performance:" << std::endl;
    std::cout << "Total time: " << total_ms << " ms for " << iterations << " iterations" << std::endl;
    std::cout << "Average time per iteration: " << avg_ms << " ms" << std::endl;
    
    float total_ops = 2.0f * B * H * L * L * D + B * H * L * L; // FMAs + exponentials
    float total_gbytes = (3 * B * H * L * D + B * H * L * L) * sizeof(float) / 1e9; // Q,K,V + scores
    
    std::cout << "FLOPS: " << (total_ops * iterations) / (total_ms / 1000) / 1e9 << " GFLOPS" << std::endl;
    std::cout << "Memory Bandwidth: " << (total_gbytes * iterations) / (total_ms / 1000) << " GB/s" << std::endl;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_scores);
    return 0;
}