// Cross-XCC coherence probe for MI350 DPX.
// One block writes iteration i into a device line. 512 blocks read it back and
// record the XCC id. A stale block saw an older value. On DPX+NPS2 this stays
// stale for the life of the process. On DPX+NPS1 it should stay at zero.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#define CK(x)                                                          \
  do {                                                                 \
    hipError_t e = (x);                                                \
    if (e != hipSuccess) {                                             \
      printf("FAIL %s -> %s line %d\n", #x, hipGetErrorString(e),      \
             __LINE__);                                                \
      exit(2);                                                         \
    }                                                                  \
  } while (0)

__global__ void wr(unsigned* buf, unsigned v, unsigned* wxcc) {
  if (threadIdx.x == 0) {
    buf[0] = v;
    *wxcc = __builtin_amdgcn_s_getreg((3 << 11) | (0 << 6) | 20);
  }
}

__global__ void rd(const unsigned* buf, unsigned* out) {
  if (threadIdx.x == 0) {
    out[2 * blockIdx.x] = buf[0];
    out[2 * blockIdx.x + 1] =
        __builtin_amdgcn_s_getreg((3 << 11) | (0 << 6) | 20);
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage: coh <iters> <gpu|h2d>\n");
    return 2;
  }
  long iters = atol(argv[1]);
  std::string mode = argv[2];
  const int B = 512;
  unsigned *buf, *out, *wx;
  CK(hipMalloc(&buf, 4096));
  CK(hipMalloc(&out, 2 * B * 4));
  CK(hipMalloc(&wx, 4));
  CK(hipMemset(buf, 0, 4096));
  std::vector<unsigned> h(2 * B), page(1024);
  long stale_l = 0, stale_b = 0;
  std::map<unsigned, long> xs, xall, wxh;
  std::map<long, long> lags;
  for (long i = 1; i <= iters; i++) {
    if (mode == "gpu") {
      wr<<<1, 64>>>(buf, (unsigned)i, wx);
    } else {
      for (auto& p : page) p = (unsigned)i;
      CK(hipMemcpy(buf, page.data(), 4096, hipMemcpyHostToDevice));
    }
    rd<<<B, 64>>>(buf, out);
    CK(hipMemcpy(h.data(), out, 2 * B * 4, hipMemcpyDeviceToHost));
    unsigned w = 9;
    if (mode == "gpu") {
      CK(hipMemcpy(&w, wx, 4, hipMemcpyDeviceToHost));
      wxh[w]++;
    }
    int sb = 0;
    for (int b = 0; b < B; b++) {
      xall[h[2 * b + 1]]++;
      if (h[2 * b] != (unsigned)i) {
        sb++;
        xs[h[2 * b + 1]]++;
        lags[(long)i - (long)h[2 * b]]++;
      }
    }
    if (sb) {
      stale_l++;
      stale_b += sb;
      if (stale_l <= 2) {
        printf("STALE iter %ld (writer xcc %u): %d blocks:", i, w, sb);
        int n = 0;
        for (int b = 0; b < B && n < 5; b++) {
          if (h[2 * b] != (unsigned)i) {
            printf(" b%d=%u(xcc%u)", b, h[2 * b], h[2 * b + 1]);
            n++;
          }
        }
        printf("\n");
      }
    }
  }
  printf("RESULT mode=%s iters=%ld stale_launches=%ld stale_blocks=%ld | readers:",
         mode.c_str(), iters, stale_l, stale_b);
  for (auto& kv : xall) printf(" xcc%u=%ld", kv.first, kv.second);
  printf(" | stale readers:");
  for (auto& kv : xs) printf(" xcc%u=%ld", kv.first, kv.second);
  printf(" | writer xcc:");
  for (auto& kv : wxh) printf(" xcc%u=%ld", kv.first, kv.second);
  printf(" | lags:");
  int n = 0;
  for (auto& kv : lags) {
    if (n++ < 6) printf(" %ld:%ld", kv.first, kv.second);
  }
  printf("\n");
  return stale_l ? 1 : 0;
}
