#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include <curand_kernel.h>
#include <curand.h>

#define SEED 921

__global__
void cu_calculate_pi_shared(double* pi_counts, curandState* rand_states, int n) 
{   
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= n) return;

    curand_init(SEED, id, 0, &rand_states[id]);

    extern __shared__ double s_prod[];
    s_prod[threadIdx.x] = 0;
    __syncthreads();

    double x, y, z;
    // Generate random (X,Y) points
    x = (double)curand_uniform (&rand_states[id]);
    y = (double)curand_uniform (&rand_states[id]);
    z = sqrt((x*x) + (y*y));

    // Check if point is in unit circle
    if (z <= 1.0)
    {
        s_prod[threadIdx.x] = 1;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        for (int thread = 0; thread < blockDim.x; thread++)
        {
            pi_counts[blockIdx.x] += s_prod[thread];
        }
    }
}

double calculate_pi_shared(int block_size, int block_count, int n) {
    // Calculate using __shared__ buffer
    curandState *dev_random;
    cudaMalloc(&dev_random, block_count*block_count*sizeof(curandState));
    size_t mem_size = sizeof(double) * block_count;
    double* d_pi_counts = 0;
    cudaMalloc(&d_pi_counts, mem_size);
    cudaMemset(d_pi_counts, 0, mem_size);
    cu_calculate_pi_shared<<<block_count, block_size, block_size>>>(d_pi_counts, dev_random, n);
    
    double* pi_counts = (double*)malloc(mem_size);
    cudaMemcpy(pi_counts, d_pi_counts, mem_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_random);
    cudaFree(d_pi_counts);

    double pi_sum = 0.0;
    for (int i = 0; i < block_count; i++) {
        pi_sum += pi_counts[i];
    }

    return 4.0 * pi_sum / n;
}

template<typename T>
__global__
void cu_calculate_pi_loop(T* pi_counts, curandState* rand_states, int n) 
{   
    int id = threadIdx.x;

    curand_init(SEED, id, 0, &rand_states[id]);

    int iterations = (n / (T)blockDim.x);
    for (int i = 0; i < iterations; i++) {
        T x, y, z;
        // Generate random (X,Y) points
        x = (T)curand_uniform (&rand_states[id]);
        y = (T)curand_uniform (&rand_states[id]);
        z = sqrt((x*x) + (y*y));

        // Check if point is in unit circle
        if (z <= 1.0)
        {
            pi_counts[id]++;
        }
    }
    pi_counts[id] /= iterations;
}

template<typename T>
T calculate_pi_loop(int block_size, int n) {
    using namespace std::chrono;
    // Calculate using Loop
    curandState *dev_random;
    cudaMalloc(&dev_random, block_size*sizeof(curandState));
    size_t mem_size = sizeof(T) * block_size;
    T* d_pi_counts = 0;
    cudaMalloc(&d_pi_counts, mem_size);
    cudaMemset(d_pi_counts, 0, mem_size);

    high_resolution_clock::time_point t_start = high_resolution_clock::now();
    cu_calculate_pi_loop<T><<<1, block_size>>>(d_pi_counts, dev_random, n);
    cudaDeviceSynchronize();
    high_resolution_clock::time_point t_end = high_resolution_clock::now();
    auto time_diff = duration_cast<duration<double>>(t_end - t_start);
    printf("Time (ms)\n");
    printf("%f\n", time_diff.count() * 1e3);
    
    T* pi_counts = (T*)malloc(mem_size);
    cudaMemcpy(pi_counts, d_pi_counts, mem_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_random);
    cudaFree(d_pi_counts);

    T pi_sum = 0.0;
    for (int i = 0; i < block_size; i++) {
        pi_sum += pi_counts[i];
    }

    return 4.0 * pi_sum / block_size;
}

int main(int argc, char* argv[])
{
    int block_size;
    int n;
    if (argc == 4){
        block_size = atoi(argv[1]);
        n = atoi(argv[2]);
    }
    else {
        printf("./bin <block_size> <num_iter> <mode>\n");
        printf("where <mode>: 0 = single precision, 1 = double precision\n");
        return;
    }
    int block_count = (n + block_size - 1) / block_size;
    printf("N: %d, Block count: %d, block size: %d\n", n, block_count, block_size);

    double pi;
    if (atoi(argv[3]) == 1)
        pi = calculate_pi_loop<double>(block_size, n);
    else 
        pi = calculate_pi_loop<float>(block_size, n);

    // Test implementations using __shared__
    //double pi2 = calculate_pi_shared(block_size, block_count, NUM_ITER);

    // Determine correctness
    printf("Result: %f, ", pi);
    if (abs(pi - 3.14159265358979323846) < 1e-4)
        printf("(error smaller than 1e-4), OK\n");
    else 
        printf("(error LARGER than 1e-4), FAIL\n");


    return 0;
}