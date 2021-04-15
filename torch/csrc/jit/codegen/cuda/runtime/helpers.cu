
__device__ constexpr int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

__device__ constexpr int alignBufferSize(int buffer, int size) {
  return (buffer + (size - 1)) & ~(size - 1);
}

__device__ double clamp(double x, double minv, double maxv) {
  return x < minv ? minv : (x > maxv ? maxv : x);
}

__device__ float clamp(float x, double minv, double maxv) {
  return x < minv ? minv : (x > maxv ? maxv : x);
}

__device__ double frac(double x) {
  return x - trunc(x);
}

__device__ float frac(float x) {
  return x - trunc(x);
}

__device__ double gelu(double x) {
  return x * normcdf(x);
}

__device__ float gelu(float x) {
  return x * normcdf(x);
}

__device__ double reciprocal(double x) {
  return 1 / x;
}

__device__ float reciprocal(float x) {
  return 1 / x;
}

__device__ double relu(double x) {
  return x <= 0 ? 0 : x;
}

__device__ float relu(float x) {
  return x <= 0 ? 0 : x;
}

__device__ double remainder(double a, double b) {
  auto mod = ::fmod(a, b);
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ float remainder(float a, float b) {
  auto mod = ::fmod(a, b);
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

__device__ float sigmoid(float x) {
  return 1 / (1 + exp(-x));
}

__device__ double threshold(double x, double t, double v) {
  return x <= t ? v : x;
}

__device__ float threshold(float x, double t, double v) {
  return x <= t ? v : x;
}

__device__ double where(bool c, double a, double b) {
  return c ? a : b;
}

__device__ float where(bool c, float a, float b) {
  return c ? a : b;
}

__device__ float where(bool c, int64_t a, int64_t b) {
  return c ? a : b;
}

__device__ double randLike(Philox rnd) {
  return uniform(rnd(), rnd());
}

__device__ float randLikef(Philox rnd) {
  return uniformf(rnd());
}

__device__ constexpr int64_t remainder(int64_t a, int64_t b) {
  auto mod = a % b;
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ constexpr int remainder(int a, int b) {
  auto mod = a % b;
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}
