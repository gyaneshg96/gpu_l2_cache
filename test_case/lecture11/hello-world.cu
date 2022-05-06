#include <stdio.h>

__global__ void print_hello() {
  printf("hello from thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

int main() {

  print_hello<<<3, 5>>>();
  cudaDeviceSynchronize();

  return 0;
}
