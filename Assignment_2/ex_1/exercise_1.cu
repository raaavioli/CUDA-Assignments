#include <stdio.h>

#define N 256

__global__
void helloKernel(int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;

    printf("Hello World! My threadId is %d\n", id);
}

int main(int argc, char** argv)
{

    helloKernel<<<1, N>>>(N);
    cudaDeviceSynchronize();
    return 0;
}