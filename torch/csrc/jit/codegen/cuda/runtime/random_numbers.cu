class Philox {
 public:
  __device__ Philox(
      unsigned long long seed,
      unsigned long long subsequence,
      unsigned long long offset) {
    key.x = (unsigned int)seed;
    key.y = (unsigned int)(seed >> 32);
    counter = make_uint4(0, 0, 0, 0);
    counter.z = (unsigned int)(subsequence);
    counter.w = (unsigned int)(subsequence >> 32);
    STATE = 0;
    incr_n(offset / 4);
  }

  __device__ unsigned long operator()() {
    if (STATE == 0) {
      uint4 counter_ = counter;
      uint2 key_ = key;
      for (int i = 0; i < 9; i++) {
        counter_ = single_round(counter_, key_);
        key_.x += (kPhilox10A);
        key_.y += (kPhilox10B);
      }
      output = single_round(counter_, key_);
      incr();
    }
    unsigned long ret = 0;
    switch (STATE) {
      case 0:
        ret = output.x;
        break;
      case 1:
        ret = output.y;
        break;
      case 2:
        ret = output.z;
        break;
      case 3:
        ret = output.w;
        break;
    }
    STATE = (STATE + 1) % 4;
    return ret;
  }

 private:
  __device__ void incr_n(unsigned long long n) {
    unsigned int nlo = (unsigned int)(n);
    unsigned int nhi = (unsigned int)(n >> 32);
    counter.x += nlo;
    if (counter.x < nlo)
      nhi++;
    counter.y += nhi;
    if (nhi <= counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }

  __device__ void incr() {
    if (++counter.x)
      return;
    if (++counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }

  __device__ unsigned int mulhilo32(
      unsigned int a,
      unsigned int b,
      unsigned int* result_high) {
    *result_high = __umulhi(a, b);
    return a * b;
  }

  __device__ uint4 single_round(uint4 ctr, uint2 key) {
    unsigned int hi0;
    unsigned int hi1;
    unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
    unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
    uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
    return ret;
  }

 private:
  static constexpr unsigned long kPhilox10A = 0x9E3779B9;
  static constexpr unsigned long kPhilox10B = 0xBB67AE85;
  static constexpr unsigned long kPhiloxSA = 0xD2511F53;
  static constexpr unsigned long kPhiloxSB = 0xCD9E8D57;

  uint4 counter = {};
  uint4 output = {};
  uint2 key = {};
  unsigned int STATE = 0;
};

__device__ float uniform(unsigned int x) {
  constexpr float kRanInvM32 = 2.3283064e-10f; // Inverse of 2^32.
  return x * kRanInvM32;
}
