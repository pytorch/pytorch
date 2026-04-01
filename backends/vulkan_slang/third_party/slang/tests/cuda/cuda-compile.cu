//TEST(smoke):COMPILE: -pass-through nvrtc -target ptx -entry hello tests/cuda/cuda-compile.cu

__global__ 
void hello(char *a, int *b) 
{
	a[threadIdx.x] += b[threadIdx.x];
}
