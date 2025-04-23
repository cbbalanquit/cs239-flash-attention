#include <iostream>
#include <cmath>
#include <memory>
#include <random>
#include <cfloat>
#include <cuda_runtime.h>

void generateMatrix(float *matrix, int n, std::mt19937 &mt)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i)
        matrix[i] = dist(mt);
}

#define TILE_Q 32
#define TILE_K 32
#define TILE_D 16

__global__ void flash_attention_kernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ output,
    int B, int H, int L, int D)
{
    const int q_idx = blockIdx.x * TILE_Q + threadIdx.x; // query index
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int d_idx = threadIdx.y; // for processing depth dimension

    if (q_idx >= L || d_idx >= D)
        return;

    const float scale = 1.0f / sqrtf((float)D);

    __shared__ float Q_tile[TILE_Q][TILE_D];
    __shared__ float K_tile[TILE_K][TILE_D];
    __shared__ float V_tile[TILE_K][TILE_D];

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc = 0.0f;

    // Load this thread's Q[q_idx] into shared memory
    if (d_idx < D && q_idx < L)
    {
        int q_offset = ((batch * H + head) * L + q_idx) * D + d_idx;
        Q_tile[threadIdx.x][d_idx] = Q[q_offset];
    }
    __syncthreads();

    for (int kt = 0; kt < L; kt += TILE_K)
    {
        int k_idx = kt + threadIdx.x;
        if (k_idx < L && d_idx < D)
        {
            int offset = ((batch * H + head) * L + k_idx) * D + d_idx;
            K_tile[threadIdx.x][d_idx] = K[offset];
            V_tile[threadIdx.x][d_idx] = V[offset];
        }
        __syncthreads();

        for (int ki = 0; ki < TILE_K && (kt + ki) < L; ++ki)
        {
            float score = 0.0f;
            for (int d = 0; d < D; ++d)
            {
                score += Q_tile[threadIdx.x][d] * K_tile[ki][d];
            }
            score *= scale;

            float m_new = fmaxf(m_i, score);
            float l_new = l_i * expf(m_i - m_new) + expf(score - m_new);
            float alpha = expf(score - m_new) / l_new;

            acc = acc * (l_i * expf(m_i - m_new) / l_new) + alpha * V_tile[ki][d_idx];
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    int out_offset = ((batch * H + head) * L + q_idx) * D + d_idx;
    output[out_offset] = acc;
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

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc((void **)&d_Q, bytes);
    cudaMalloc((void **)&d_K, bytes);
    cudaMalloc((void **)&d_V, bytes);
    cudaMalloc((void **)&d_O, bytes);

    cudaMemcpy(d_Q, h_Q.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.get(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.get(), bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(TILE_Q, TILE_D);
    dim3 grid(ceil((float)L / TILE_Q), H, B);

    flash_attention_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, B, H, L, D);

    cudaMemcpy(h_O.get(), d_O, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Sample output[0][0][0][:5]: ";
    for (int i = 0; i < 5; ++i)
        std::cout << h_O[i] << " ";
    std::cout << std::endl;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return 0;
}
