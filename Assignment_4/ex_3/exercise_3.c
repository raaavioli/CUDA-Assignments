// Template file for the OpenCL Assignment 4

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct timeval tval;

/**
 * Calculates the elapsed time between two time intervals (in milliseconds).
 */
double get_elapsed(tval t0, tval t1)
{
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L + (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}

void saxpy (float* X, float* Y, int a, int n) {
  for (int i = 0; i < n; i++)
    Y[i] += X[i] * a;
}

void openACC_saxpy (float* restrict X, float* restrict Y, int a, int n) {
#pragma acc parallel loop
  for (int i = 0; i < n; i++)
    Y[i] = Y[i] + X[i] * a;
}

int main(int argc, char *argv[]) {
  tval times[2] = { 0 };
  double elapsed[2] = { 0 };

  int array_count = 1e4;
  if (argc == 2)
    array_count = atoi(argv[1]);
  int a = 2.0f;
  float* X, *Y;
  X = malloc(array_count * sizeof(float));
  Y = malloc(array_count * sizeof(float));
  for (int i = 0; i < array_count; i++)
    X[i] = Y[i] = 2.0f;

  // CPU saxpy
  printf("Computing SAXPY on the CPU... ");
  gettimeofday(&times[0], NULL);
  saxpy(X, Y, a, array_count);
  gettimeofday(&times[1], NULL);
  elapsed[0] = get_elapsed(times[0], times[1]);
  printf("Done!\n");

  // OpenACC saxpy
  printf("Computing SAXPY on the OpenACC... ");
  gettimeofday(&times[0], NULL);
  openACC_saxpy(X, Y, a, array_count);
  gettimeofday(&times[1], NULL);
  elapsed[1] = get_elapsed(times[0], times[1]);
  printf("Done!\n");

  printf("OpenACC (ms)\tCPU (ms)\n");
  printf("%f\t%f\n", elapsed[1], elapsed[0]);

  free(Y);
  free(X);
  
  return 0;
}

