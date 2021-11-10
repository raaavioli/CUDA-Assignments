#include <stdio.h>
#include <chrono>

#define ARRAY_SIZE 10000000
#define BLOCK_SIZE 256

__host__
void saxpy(int n, float a, float* x, float* y)
{
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

__global__
void cu_saxpy(int n, float a, float* x, float* y)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;

    y[id] = a * x[id] + y[id];
}

int main(int argc, char** argv)
{
    using namespace std::chrono;

    // Setup
    float a = 2.0f;
    float* h_x = (float*) malloc(sizeof(float) * ARRAY_SIZE);
    float* h_y = (float*) malloc(sizeof(float) * ARRAY_SIZE);
    for (int i = 0; i < ARRAY_SIZE; i++)
        h_x[i] = h_y[i] = 2.0;
    float* d_x = 0;
    float* d_y = 0;
    cudaMalloc(&d_x, sizeof(float) * ARRAY_SIZE);
    cudaMalloc(&d_y, sizeof(float) * ARRAY_SIZE);



    // --------- Device SAXPY --------
    printf("Computing SAXPY on the GPU… ");
    high_resolution_clock::time_point t_cu_mem_saxpy_start = high_resolution_clock::now();
    cudaMemcpy(d_x, h_x, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);
    int blocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    high_resolution_clock::time_point t_cu_saxpy_start = high_resolution_clock::now();
    cu_saxpy<<<blocks, BLOCK_SIZE>>>(ARRAY_SIZE, a, d_x, d_y);
    cudaDeviceSynchronize();
    high_resolution_clock::time_point t_cu_saxpy_end = high_resolution_clock::now();

    float* d_y_res = (float*) malloc(sizeof(float) * ARRAY_SIZE);
    cudaMemcpy(d_y_res, d_y, sizeof(float) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
    high_resolution_clock::time_point t_cu_mem_saxpy_end = high_resolution_clock::now();
    printf("Done!\n");
    // -------- End Device SAXPY ------



    // ---------- Host SAXPY ---------
    printf("Computing SAXPY on the CPU… ");
    high_resolution_clock::time_point t_saxpy_start = high_resolution_clock::now();
    saxpy(ARRAY_SIZE, a, h_x, h_y);
    high_resolution_clock::time_point t_saxpy_end = high_resolution_clock::now();
    printf("Done!\n");
    // --------- End Host SAXPY -----

    // Correctness comparison
    bool equal = true;
    printf("Comparing the output for each implementation… ");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (abs(d_y_res[i] - h_y[i]) > 1e-5)
            equal = false;
    }

    if (equal)
        printf("Correct!\n");
    else 
        printf("!!!!! Incorrect !!!!!\n");

    // Timing comparison
    duration<double> saxpy_time = duration_cast<duration<double>>(t_saxpy_end - t_saxpy_start);
    duration<double> cu_saxpy_time = duration_cast<duration<double>>(t_cu_saxpy_end - t_cu_saxpy_start);
    duration<double> cu_mem_saxpy_time = duration_cast<duration<double>>(t_cu_mem_saxpy_end - t_cu_mem_saxpy_start);
    printf("Timing\n");
    printf("saxpy (ms)\tcu_saxpy (ms)\tcu_saxpy + mem (ms)\n");
    printf("%f\t%f\t%f\n", saxpy_time.count() * 1e3, cu_saxpy_time.count() * 1e3, cu_mem_saxpy_time.count() * 1e3);

    return 0;
}