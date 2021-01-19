__device__ constexpr int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

__device__ constexpr int alignBufferSize(int buffer, int size) {
  return (buffer + (size - 1)) & ~(size - 1);
}

__device__ float clamp(float x, float minv, float maxv) {
  return x < minv ? minv : (x > maxv ? maxv : x);
}

__device__ float frac(float x) {
  return x - truncf(x);
}

__device__ float gelu(float x) {
  return x * normcdf(x);
}

__device__ float reciprocal(float x) {
  return 1.f / x;
}

__device__ float relu(float x) {
  return x <= 0.f ? 0.f : x;
}

__device__ float remainder(float a, float b) {
  return a - b * floorf(a / b);
}

__device__ float sigmoid(float x) {
  return 1.f / (1.f + expf(-x));
}

__device__ float threshold(float x, float t, float v) {
  return x <= t ? v : x;
}

__device__ float where(bool c, float a, float b) {
  return c ? a : b;
}

__device__ float randLike(Philox rnd) {
  return uniform(rnd());
}
