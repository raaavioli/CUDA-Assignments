#include <stdio.h>
#include <chrono>

#define PI 3.14159265358
#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 1000

struct Particle {
    float3 position;
    float3 velocity;
};

__host__
__device__
float3 operator+ (float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__
void cu_move_particles(Particle* particles, int num_particles, int iteration, int num_iterations) 
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_particles) return;

    float angle = PI * 2.0f * (id / (float) num_particles) * (iteration / (float) num_iterations);
    float3 dv = make_float3(sin(angle), cos(angle), sin(angle));
    particles[id].velocity = particles[id].velocity + dv;
    particles[id].position = particles[id].position + particles[id].velocity;
}

__host__
void move_particles(Particle* particles, int num_particles, int iteration, int num_iterations)
{
    for (int i = 0; i < num_particles; i++)
    {
        float angle = PI * 2.0f * (i / (float) num_particles) * (iteration / (float) num_iterations);
        float3 dv = make_float3(sin(angle), cos(angle), sin(angle));
        particles[i].velocity = particles[i].velocity + dv;
        particles[i].position = particles[i].position + particles[i].velocity;
    }
}

int main(int argc, char** argv)
{
    using namespace std::chrono;

    int num_particles = NUM_PARTICLES;
    int num_iterations = NUM_ITERATIONS;
    if (argc == 3) {
        num_particles = atoi(argv[1]);
        num_iterations = atoi(argv[2]);
        printf("Running simulation...\nnum particles: %d\nnum iterations: %d\n", num_particles, num_iterations);
    } else {
        printf("./bin <num_particles> <num_iterations>\n");
        return;
    }

    int buffer_size = sizeof(Particle) * num_particles;

    Particle* particles = (Particle*) malloc(buffer_size);
    for (int i = 0; i < num_particles; i++)
    {
        particles[i].velocity = {0};
        particles[i].position = {0};
    }

    // CPU Simulation
    printf("Simulating particles on CPU… ");
    high_resolution_clock::time_point t_cpu_start = high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) 
    {
        move_particles(particles, num_particles, i, num_iterations);
    }
    high_resolution_clock::time_point t_cpu_end = high_resolution_clock::now();
    printf("Done!\n");

    // GPU Simulation
    printf("Simulating particles on GPU… block-size: ");
    Particle* cu_particles = 0;
    cudaMalloc(&cu_particles, buffer_size);
    Particle* cu_particles_res = (Particle*) malloc(buffer_size);
    const int block_iterations = 5;
    duration<double> cu_durations_per_blocksize[block_iterations];
    duration<double> cu_mem_durations_per_blocksize[block_iterations];
    int block_size = 16;
    for (int i = 0; i < 5; i++) {
        printf("%d..., ", block_size);
        high_resolution_clock::time_point t_cu_mem_start = high_resolution_clock::now();
        cudaMemset(cu_particles, 0, buffer_size);
        int num_blocks = (num_particles + block_size - 1) / block_size;
        high_resolution_clock::time_point t_cu_start = high_resolution_clock::now();
        for (int j = 0; j < num_iterations; j++) 
        {
            cu_move_particles<<<num_blocks, block_size>>>(cu_particles, num_particles, j, num_iterations);
            // cudaDeviceSynchronize should not be needed here as each thread within a block operates only
            // on the same memory region, and will continue to the next once finished with previous iteration
        }
        cudaDeviceSynchronize();
        high_resolution_clock::time_point t_cu_end = high_resolution_clock::now();
        cu_durations_per_blocksize[i] = duration_cast<duration<double>>(t_cu_end - t_cu_start);
        block_size *= 2;

        cudaMemcpy(cu_particles_res, cu_particles, buffer_size, cudaMemcpyDeviceToHost);
        high_resolution_clock::time_point t_cu_mem_end = high_resolution_clock::now();
        cu_mem_durations_per_blocksize[i] = duration_cast<duration<double>>(t_cu_mem_end - t_cu_mem_start);
        cudaDeviceSynchronize();
    }
    printf("Done!!\n");

    // Equality comparison
    bool equal = true;
    printf("Comparing the output for each implementation… ");
    for (int i = 0; i < num_particles; i++)
    {
        // Divide difference by the minimum magnitude to get a fair comparison for large and 
        // small values having some rounding error in the least significant bits
        float minx = min(cu_particles_res[i].position.x, particles[i].position.x);
        float miny = min(cu_particles_res[i].position.y, particles[i].position.y);
        float minz = min(cu_particles_res[i].position.z, particles[i].position.z);
        if (abs((cu_particles_res[i].position.x - particles[i].position.x) / minx) > 1e-6 &&
            abs((cu_particles_res[i].position.y - particles[i].position.y) / miny) > 1e-6 && 
            abs((cu_particles_res[i].position.z - particles[i].position.z) / minz) > 1e-6 ) 
        {
            equal = false;
            printf("\ngpu: {%f, %f, %f} \ncpu: {%f, %f, %f}\n",    
                cu_particles_res[i].position.x, cu_particles_res[i].position.y, cu_particles_res[i].position.z, 
                particles[i].position.x, particles[i].position.y, particles[i].position.z);
            break;
        }
    }
    if (equal)
        printf("Correct!\n");
    else 
        printf("!!!!! Incorrect !!!!!\n");


    // Timing comparison
    duration<double> cpu_time = duration_cast<duration<double>>(t_cpu_end - t_cpu_start);
    
    printf("cpu (ms)\tcu16 (ms)\tcu32 (ms)\tcu64 (ms)\tcu128 (ms)\tcu256 (ms)\n");
    printf("%f\t", cpu_time.count() * 1e3);
    for (int i = 0; i < 5; i++) {
        //printf("%f\t%f\t", cu_durations_per_blocksize[i].count() * 1e3, cu_mem_durations_per_blocksize[i].count() * 1e3);
        printf("%f\t", cu_durations_per_blocksize[i].count() * 1e3);
    }
    printf("\n");

    cudaFree(cu_particles);

    return 0;
}