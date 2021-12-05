#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <CL/cl.h>
#include <math.h>
#include <string.h>

typedef struct timeval tval;

#define PI 3.14159265358
#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 1000


// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

// This is a macro for checking the error variable.
#define CHK_ERROR(err) \
  if ((err) != CL_SUCCESS) {\
    fprintf(stderr,"Error %d: %s\n", __LINE__, clGetErrorString(err)); \
    exit(0); \
  }

/**
 * Calculates the elapsed time between two time intervals (in milliseconds).
 */
double get_elapsed(tval t0, tval t1)
{
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L + (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}

struct Particle {
    float position[3];
    float velocity[3];
};

typedef struct Particle Particle_t;


/*void move_particles(Particle* particles, int num_particles, int num_iterations, int iteration) 
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_particles) return;

    float angle = PI * 2.0f * (id / (float) num_particles) * (iteration / (float) num_iterations);
    float3 dv = make_float3(sin(angle), cos(angle), sin(angle));
    particles[id].velocity = particles[id].velocity + dv;
    particles[id].position = particles[id].position + particles[id].velocity;
}*/

const char *move_particles_src = 
"typedef struct Particle { \
    float position[3]; \
    float velocity[3]; \
} Particle_t; \
__kernel \
void move_particles (__global Particle_t* particles, int num_particles, int num_iterations, int iteration) { \
    int gid = get_global_id(0); \
    if (gid >= num_particles) return; \
    float angle = 3.14159265358f * 2.0f * (gid / (float) num_particles) * (iteration / (float) num_iterations); \
    float dx = sin(angle); \
    float dy = cos(angle); \
    float dz = sin(angle); \
    particles[gid].velocity[0] = particles[gid].velocity[0] + dx; \
    particles[gid].velocity[1] = particles[gid].velocity[1] + dy; \
    particles[gid].velocity[2] = particles[gid].velocity[2] + dz; \
    particles[gid].position[0] = particles[gid].position[0] + particles[gid].velocity[0]; \
    particles[gid].position[1] = particles[gid].position[1] + particles[gid].velocity[1]; \
    particles[gid].position[2] = particles[gid].position[2] + particles[gid].velocity[2]; \
} ";

void move_particles(Particle_t* particles, int num_particles, int iteration, int num_iterations)
{
    for (int i = 0; i < num_particles; i++)
    {
        float angle = PI * 2.0f * (i / (float) num_particles) * (iteration / (float) num_iterations);
        float dx = sin(angle);
        float dy = cos(angle); 
        float dz = sin(angle);
        particles[i].velocity[0] = particles[i].velocity[0] + dx;
        particles[i].velocity[1] = particles[i].velocity[1] + dy;
        particles[i].velocity[2] = particles[i].velocity[2] + dz;
        particles[i].position[0] = particles[i].position[0] + particles[i].velocity[0];
        particles[i].position[1] = particles[i].position[1] + particles[i].velocity[1];
        particles[i].position[2] = particles[i].position[2] + particles[i].velocity[2];
    }
}

struct CLData {
    cl_platform_id* platforms; 
    cl_uint n_platform;
    cl_device_id* device_list;
    cl_uint n_devices;
    cl_context context;
    cl_command_queue cmd_queue;

    cl_program program;
    cl_kernel kernel;

    cl_mem particles;
};

typedef struct CLData CLData_t;

void InitCL(CLData_t* clData, Particle_t* cpu_particles, int num_particles, int num_iterations) 
{
    CHK_ERROR(clGetPlatformIDs(0, NULL, &clData->n_platform));
    clData->platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*clData->n_platform);
    CHK_ERROR(clGetPlatformIDs(clData->n_platform, clData->platforms, NULL));

    // Find and sort devices
    CHK_ERROR(clGetDeviceIDs( clData->platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &clData->n_devices));
    clData->device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*clData->n_devices);
    CHK_ERROR(clGetDeviceIDs( clData->platforms[0],CL_DEVICE_TYPE_GPU, clData->n_devices, clData->device_list, NULL));
    
    // Create and initialize an OpenCL context
    cl_int err;
    clData->context = clCreateContext( NULL, clData->n_devices, clData->device_list, NULL, NULL, &err);
    CHK_ERROR(err);

    // Create a command queue
    clData->cmd_queue = clCreateCommandQueue(clData->context, clData->device_list[0], 0, &err);
    CHK_ERROR(err);

    clData->particles = clCreateBuffer(clData->context, CL_MEM_READ_ONLY, num_particles * sizeof(Particle_t), NULL, &err); 
    CHK_ERROR(err);

    CHK_ERROR(clEnqueueWriteBuffer(clData->cmd_queue, clData->particles, CL_TRUE, 0, num_particles * sizeof(Particle_t), cpu_particles, 0, NULL, NULL));

    // Create an OpenCL program (cl_program)
    clData->program = clCreateProgramWithSource(clData->context, 1, &move_particles_src, NULL, &err);
    CHK_ERROR(err);

    // Build/compile the program (cCreateKernel)
    err = clBuildProgram(clData->program, 1, clData->device_list, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    { 
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(clData->program, clData->device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr,"Build error: %s\n", buffer); 
        exit(0); 
    }

    // Create a kernel (clCreateKernel)
    // Particle* particles, int num_particles, int num_iterations, int iteration
    clData->kernel = clCreateKernel(clData->program, "move_particles", &err);
    CHK_ERROR(clSetKernelArg(clData->kernel, 0, sizeof(cl_mem), (void*) &clData->particles));
    CHK_ERROR(clSetKernelArg(clData->kernel, 1, sizeof(int), &num_particles));
    CHK_ERROR(clSetKernelArg(clData->kernel, 2, sizeof(int), &num_iterations));
}

int main(int argc, char** argv)
{
    tval times[2] = { 0 };
    double elapsed[6] = { 0 };
    int num_particles = NUM_PARTICLES;
    int num_iterations = NUM_ITERATIONS;
    if (argc == 3) {
        num_particles = atoi(argv[1]);
        num_iterations = atoi(argv[2]);
        printf("Running simulation...\nnum particles: %d\nnum iterations: %d\n", num_particles, num_iterations);
    } else {
        printf("./bin <num_particles> <num_iterations>\n");
        return 0;
    }

    int buffer_size = sizeof(Particle_t) * num_particles;

    Particle_t* particles = (Particle_t*) malloc(buffer_size);
    Particle_t* dump = (Particle_t*) malloc(buffer_size);
    memset(particles, 0, buffer_size);
    // Initialize OpenCL pipeline
    CLData_t clData;
    InitCL(&clData, particles, num_particles, num_iterations);

    // CPU Simulation
    printf("Simulating particles on CPU… ");
    gettimeofday(&times[0], NULL);
    for (int i = 0; i < num_iterations; i++) 
    {
        move_particles(particles, num_particles, i, num_iterations);
    }
    gettimeofday(&times[1], NULL);
    elapsed[0] = get_elapsed(times[0], times[1]);
    printf("Done!\n");

    // OpenCL Simulation

    cl_int err;
    printf("Simulating particles on GPU… block-size: ");
    
    // Invoke the kernel (clEnqueueNDRangeKernel)
    size_t local_work_size = 16;
    printf("Computing SAXPY on the GPU... \n");
    printf("OpenCL kernel settings: \n");
    Particle_t* cl_particles_res = (Particle_t*) malloc(buffer_size);
    for (int i = 0; i < 5; i++) {
        size_t global_work_size = num_particles + (local_work_size - num_particles % local_work_size);
        printf("\tArray size: %d, Work items: %d, Work groups: %d, Work group size: %d\n", 
            num_particles, (int) global_work_size, (int) (global_work_size / local_work_size), (int) local_work_size);
        gettimeofday(&times[0], NULL);
        for (int j = 0; j < num_iterations; j++) {
            CHK_ERROR(clSetKernelArg(clData.kernel, 3, sizeof(int), &j));
            CHK_ERROR(clEnqueueNDRangeKernel(clData.cmd_queue, clData.kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));

            // Wait for all commands in the queue to finish (clFinish)
            CHK_ERROR(clFlush(clData.cmd_queue));
            CHK_ERROR(clFinish(clData.cmd_queue));
        }

        // Store first iteration into cl_particles_res to be able to compare with cpu-versions
        if (i == 0) {
            CHK_ERROR(clEnqueueReadBuffer(clData.cmd_queue, clData.particles, CL_TRUE, 0, num_particles * sizeof(Particle_t), cl_particles_res, 0, NULL, NULL));
        }
        else {
            CHK_ERROR(clEnqueueReadBuffer(clData.cmd_queue, clData.particles, CL_TRUE, 0, num_particles * sizeof(Particle_t), dump, 0, NULL, NULL));
        } 
        
        gettimeofday(&times[1], NULL);
        elapsed[1 + i] = get_elapsed(times[0], times[1]);
        local_work_size *= 2;
    }
    printf("Done!!\n");

    // Equality comparison
    int equal = 1;
    printf("Comparing the output for each implementation… ");
    for (int i = 0; i < num_particles; i++)
    {
        // Divide difference by the minimum magnitude to get a fair comparison for large and 
        // small values having some rounding error in the least significant bits
        float minx = fmin(cl_particles_res[i].position[0], particles[i].position[0]);
        float miny = fmin(cl_particles_res[i].position[1], particles[i].position[1]);
        float minz = fmin(cl_particles_res[i].position[2], particles[i].position[2]);
        if (abs((cl_particles_res[i].position[0] - particles[i].position[0]) / minx) > 1e-6 &&
            abs((cl_particles_res[i].position[1] - particles[i].position[1]) / miny) > 1e-6 && 
            abs((cl_particles_res[i].position[2] - particles[i].position[2]) / minz) > 1e-6 ) 
        {
            equal = 0;
            printf("\ngpu: {%f, %f, %f} \ncpu: {%f, %f, %f}\n",    
                cl_particles_res[i].position[0], cl_particles_res[i].position[1], cl_particles_res[i].position[2], 
                particles[i].position[0], particles[i].position[1], particles[i].position[2]);
            break;
        }
    }
    if (equal)
        printf("Correct!\n");
    else 
        printf("!!!!! Incorrect !!!!!\n");


    // Timing comparison    
    printf("cpu (ms)\tcu16 (ms)\tcu32 (ms)\tcu64 (ms)\tcu128 (ms)\tcu256 (ms)\n");
    printf("%f\t", elapsed[0]);
    for (int i = 0; i < 5; i++) {
        printf("%f\t", elapsed[1 + i]);
    }
    printf("\n");

    free(cl_particles_res);
    free(dump);
    free(particles);

    free(clData.platforms);
    free(clData.device_list);

    return 0;
}


// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clGetErrorString(int errorCode) {
  switch (errorCode) {
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  case -69: return "CL_INVALID_PIPE_SIZE";
  case -70: return "CL_INVALID_DEVICE_QUEUE";
  case -71: return "CL_INVALID_SPEC_ID";
  case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
  case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
  case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
  case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
  case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
  case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
  case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
  case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
  case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
  case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
  case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
  case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
  case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
  case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
  case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
  case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
  case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
  case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
  default: return "CL_UNKNOWN_ERROR";
  }
}