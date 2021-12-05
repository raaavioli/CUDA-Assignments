// Template file for the OpenCL Assignment 4

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <CL/cl.h>

typedef struct timeval tval;

// This is a macro for checking the error variable.
#define CHK_ERROR(err) \
  if ((err) != CL_SUCCESS) \
    fprintf(stderr,"Error: %s\n", clGetErrorString(err)); 

// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

/**
 * Calculates the elapsed time between two time intervals (in milliseconds).
 */
double get_elapsed(tval t0, tval t1)
{
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L + (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}

const char *mykernel = 
"__kernel \
void saxpy (__global float* X, __global float* Y, int a, int n) { \
  int gid = get_global_id(0); \
  if (gid >= n) return; \
  Y[gid] = Y[gid] + a * X[gid]; \
} ";

void saxpy (float* X, float* Y, int a, int n) {
  for (int i = 0; i < n; i++)
    Y[i] += X[i] * a;
}

void openACC_saxpy (float* X, float* Y, int a, int n) {
  #pragma acc parallel loop {
    for (int i = 0; i < n; i++)
      Y[i] += X[i] * a;
  }
}

int main(int argc, char *argv[]) {
  cl_platform_id* platforms; 
  cl_uint n_platform;
  cl_int err;
  tval times[2] = { 0 };
  double elapsed[3] = { 0 };

  // Find OpenCL Platforms
  CHK_ERROR(clGetPlatformIDs(0, NULL, &n_platform));
  platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
  CHK_ERROR(clGetPlatformIDs(n_platform, platforms, NULL));

  // Find and sort devices
  cl_device_id* device_list; 
  cl_uint n_devices;
  CHK_ERROR(clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices));
  device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
  CHK_ERROR(clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL));
  
  // Create and initialize an OpenCL context
  cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);
  CHK_ERROR(err);

  // Create a command queue
  cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);
  CHK_ERROR(err);

  int array_count = 1e4;
  if (argc == 2)
    array_count = atoi(argv[1]);
  int a = 2.0f;
  float* X, *Y, *Y_res;
  X = malloc(array_count * sizeof(float));
  Y = malloc(array_count * sizeof(float));
  Y_res = malloc(array_count * sizeof(float));
  for (int i = 0; i < array_count; i++)
    X[i] = Y[i] = 2.0f;

  // CPU saxpy
  printf("Computing SAXPY on the CPU... ");
  gettimeofday(&times[0], NULL);
  saxpy(Y, X, a, array_count);
  gettimeofday(&times[1], NULL);
  elapsed[0] = get_elapsed(times[0], times[1]);
  printf("Done!\n");

  // OpenACC saxpy
  printf("Computing SAXPY on the OpenACC... ");
  gettimeofday(&times[0], NULL);
  saxpy(Y, X, a, array_count);
  gettimeofday(&times[1], NULL);
  elapsed[2] = get_elapsed(times[0], times[1]);
  printf("Done!\n");

  // OpenCL saxpy
  cl_mem X_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, array_count * sizeof(float), NULL, &err); 
  CHK_ERROR(err);
  cl_mem Y_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, array_count * sizeof(float), NULL, &err); 
  CHK_ERROR(err);

  CHK_ERROR(clEnqueueWriteBuffer(cmd_queue, X_dev, CL_TRUE, 0, array_count * sizeof(float), X, 0, NULL, NULL));
  CHK_ERROR(clEnqueueWriteBuffer(cmd_queue, Y_dev, CL_TRUE, 0, array_count * sizeof(float), Y, 0, NULL, NULL));

  // Create an OpenCL program (cl_program)
  cl_program program = clCreateProgramWithSource(context, 1, &mykernel, NULL, &err);
  CHK_ERROR(err);

  // Build/compile the program (cCreateKernel)
  err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  { 
    size_t len;
    char buffer[2048];
    clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    fprintf(stderr,"Build error: %s\n", buffer); return 0; 
  }

  // Create a kernel (clCreateKernel)
  cl_kernel kernel = clCreateKernel(program, "saxpy", &err);
  CHK_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &X_dev));
  CHK_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &Y_dev));
  CHK_ERROR(clSetKernelArg(kernel, 2, sizeof(int), &a));
  CHK_ERROR(clSetKernelArg(kernel, 3, sizeof(int), &array_count));

  // Invoke the kernel (clEnqueueNDRangeKernel)
  size_t local_work_size = 256;
  size_t global_work_size = array_count + (local_work_size - array_count % local_work_size);
  printf("OpenCL kernel settings: \n\tArray size: %d, Work items: %d, Work groups: %d, Work group size: %d\n", 
    array_count, (int) global_work_size, (int) (global_work_size / local_work_size), (int) local_work_size);
  printf("Computing SAXPY on the GPU... ");
  gettimeofday(&times[0], NULL);
  CHK_ERROR(clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
  printf("Done!\n");

  // Wait for all commands in the queue to finish (clFinish)
  CHK_ERROR(clFlush(cmd_queue));
  CHK_ERROR(clFinish(cmd_queue));
  gettimeofday(&times[1], NULL);
  elapsed[1] = get_elapsed(times[0], times[1]);
  
  CHK_ERROR(clEnqueueReadBuffer(cmd_queue, Y_dev, CL_TRUE, 0, array_count * sizeof(float), Y_res, 0, NULL, NULL));


  // Check correctness
  int correct = 1;
  printf("Comparing the output for each implementation…");
  for (int i = 0; i < array_count; i++)
  {
    if ((Y_res[i] - (a * X[i] + Y[i])) > 1e-3)
    {
      printf("\nError at i = %d: Y != a * X + Y. | %f != %f \n", i, Y_res[i], (a * X[i] + Y[i]));
      correct = 0;
      break;
    }
  }

  if (correct)
    printf(" Correct!\n");

  printf("GPU (ms)\tCPU (ms)\tOpenACC\n");
  printf("%f\t%f\t%f\n", elapsed[1], elapsed[0], elapsed[2]);
  
  // Finally, release all that we have allocated.
  CHK_ERROR(clReleaseCommandQueue(cmd_queue));
  CHK_ERROR(clReleaseContext(context));

  free(device_list);
  free(platforms);
  free(Y_res);
  free(Y);
  free(X);
  
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
