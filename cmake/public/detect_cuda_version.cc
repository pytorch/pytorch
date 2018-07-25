#include <cuda.h>
#include <cstdio>
int main() {
  printf("%d.%d", CUDA_VERSION / 1000, (CUDA_VERSION / 10) % 100);
  return 0;
}
