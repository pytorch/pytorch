#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/ceil_div.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/FFT.h>
#include <algorithm>
#include <numeric>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fft_c2c.h>
#include <ATen/ops/_fft_c2c_native.h>
#include <ATen/ops/_fft_c2r_native.h>
#include <ATen/ops/_fft_r2c_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/polar.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {
namespace {
MPSGraphFFTScalingMode normalization_to_ScalingMode(int64_t normalization) {
  switch (static_cast<fft_norm_mode>(normalization)) {
    case fft_norm_mode::none:
      return MPSGraphFFTScalingModeNone;
    case fft_norm_mode::by_n:
      return MPSGraphFFTScalingModeSize;
    case fft_norm_mode::by_root_n:
      return MPSGraphFFTScalingModeUnitary;
    default:
      break;
  }
  TORCH_CHECK(false, "Unsupported normalization type", normalization);
}

NSArray<NSNumber*>* IntArrayToNSArray(IntArrayRef arr) {
  auto rc = [NSMutableArray<NSNumber*> arrayWithCapacity:arr.size()];
  for (const auto idx : c10::irange(arr.size())) {
    rc[idx] = [NSNumber numberWithInteger:arr[idx]];
  }
  return rc;
}

// MPSGraph FFT can only transform axes within the last four dimensions, so when
// an axis falls outside that window we permute it into the trailing dims, transform, then invert.
struct FFTAxisPlan {
  bool needsTranspose = false;
  NSArray<NSNumber*>* axes = nil;
  NSArray<NSNumber*>* permutation = nil;
  NSArray<NSNumber*>* inversePermutation = nil;
};

FFTAxisPlan computeFFTAxisPlan(IntArrayRef dim, int64_t ndim) {
  TORCH_CHECK(static_cast<int64_t>(dim.size()) <= 4,
              "MPS FFT only supports transforming up to 4 dimensions, but got ",
              dim.size(),
              " dimensions");
  FFTAxisPlan plan;
  if (std::all_of(dim.begin(), dim.end(), [&](int64_t d) { return d >= ndim - 4; })) {
    plan.axes = IntArrayToNSArray(dim);
    return plan;
  }

  plan.needsTranspose = true;
  const auto isTransformDim = at::dim_list_to_bitset(dim, ndim);
  std::vector<int64_t> perm;
  perm.reserve(ndim);
  for (const auto i : c10::irange(ndim)) {
    if (!isTransformDim[i]) {
      perm.push_back(i);
    }
  }
  perm.insert(perm.end(), dim.begin(), dim.end());
  std::vector<int64_t> inverse(ndim);
  for (const auto i : c10::irange(ndim)) {
    inverse[perm[i]] = i;
  }
  std::vector<int64_t> remappedAxes;
  remappedAxes.reserve(dim.size());
  for (const auto i : c10::irange(ndim - static_cast<int64_t>(dim.size()), ndim)) {
    remappedAxes.push_back(i);
  }
  plan.axes = IntArrayToNSArray(remappedAxes);
  plan.permutation = IntArrayToNSArray(perm);
  plan.inversePermutation = IntArrayToNSArray(inverse);
  return plan;
}

MPSGraphTensor* applyFFTInputPlan(MPSGraph* mpsGraph, MPSGraphTensor* input, const FFTAxisPlan& plan) {
  if (!plan.needsTranspose) {
    return input;
  }
  return [mpsGraph transposeTensor:input permutation:plan.permutation name:nil];
}

MPSGraphTensor* applyFFTOutputPlan(MPSGraph* mpsGraph, MPSGraphTensor* output, const FFTAxisPlan& plan) {
  if (!plan.needsTranspose) {
    return output;
  }
  return [mpsGraph transposeTensor:output permutation:plan.inversePermutation name:nil];
}

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/FFT_metallib.h>
#endif

// Stage radices of the threadgroup kernels (the kFFTMaskP2357 set) and of the global-memory passes, largest first
constexpr auto kRadices = {16, 8, 4, 2, 3, 5, 7};
#define FFT_RADIX_ENTRY(R) R,
constexpr auto kGlobalRadices = {FFT_GLOBAL_RADICES(FFT_RADIX_ENTRY)};
constexpr auto kFixedSizes = {FFT_FIXED_SIZES(FFT_RADIX_ENTRY)};
constexpr auto kContigSizes = {FFT_CONTIG_SIZES(FFT_RADIX_ENTRY)};
#undef FFT_RADIX_ENTRY
// Below this many elements, extra launches cost more than they save
constexpr int64_t kLongPoints = 1 << 17;
// Past this length, tiles strided on both sides no longer coalesce
constexpr int64_t kMaxCoalescedPoints = 1024;

// Radix sequence of n over `rs` (largest first; {1} for n = 1), empty when n has other prime factors
c10::SmallVector<uint32_t, kFFTMaxStages> radices(int64_t n, std::initializer_list<int> rs) {
  c10::SmallVector<uint32_t, kFFTMaxStages> plan(n == 1, 1);
  for (const auto r : rs) {
    for (; n % r == 0; n /= r) {
      plan.push_back(r);
    }
  }
  return n == 1 ? plan : decltype(plan){};
}

bool is_smooth(int64_t n) {
  return !radices(n, kRadices).empty();
}

// Cached MPS tables. W: W_n^k for k < n. W2, for long axes: W_n^b for b < 4096 then W_n^(4096 a). Chirp: Bluestein's
// exp(-+ i pi k^2 / n), and ChirpFFT: the M-point FFT of its conjugate wrapped around zero, M a power of two >= 2n - 1
enum class Table { W, W2, Chirp, ChirpFFT };

Tensor table(Table kind, int64_t n, bool inverse = false) {
  static std::mutex mutex;
  static std::map<std::tuple<Table, int64_t, bool>, Tensor> cache;
  std::lock_guard<std::mutex> guard(mutex);
  auto& t = cache[{kind, n, inverse}];
  if (t.defined()) {
    return t;
  }
  const bool chirp = kind == Table::Chirp || kind == Table::ChirpFFT;
  const auto k = kind == Table::W2
      ? at::cat({at::arange(kFFTTwiddleBlock, at::kDouble), at::arange(0, n, kFFTTwiddleBlock, at::kDouble)})
      : at::arange(n, at::kDouble);
  const auto angle = chirp ? k.mul(k).remainder(2 * n).mul((inverse ? 1 : -1) * M_PI / n) : k.mul(-2 * M_PI / n);
  const auto c = at::polar(at::ones_like(angle), angle);
  const auto M = static_cast<int64_t>(std::bit_ceil<uint64_t>(2 * n - 1));
  const auto b = kind == Table::ChirpFFT
      ? at::fft_fft(at::cat({c.conj(), at::zeros({M - 2 * n + 1}, c.options()), c.narrow(0, 1, n - 1).conj().flip(0)}))
      : c;
  return t = b.to(at::device(kMPS).dtype(kComplexFloat));
}

struct Dim {
  int64_t size;
  int64_t in_stride;
  int64_t out_stride;
};

struct Batch {
  Tensor in;
  Tensor out;
  c10::SmallVector<Dim, c10::metal::max_ndim> dims; // Innermost first
  // Transforms, and the element strides along the axis
  int64_t numel = 1;
  int64_t si = 0;
  int64_t so = 0;
};

// Batch dims of a pass along `dim` (every other dim), innermost first, merged where both tensors allow it
Batch collapse(const Tensor& in, const Tensor& out, int64_t dim) {
  c10::SmallVector<int64_t, c10::metal::max_ndim> dims;
  for (const auto d : c10::irange(in.dim())) {
    if (d != dim && in.size(d) != 1) {
      dims.push_back(d);
    }
  }
  std::sort(dims.begin(), dims.end(), [&](auto a, auto b) {
    return std::make_pair(std::abs(in.stride(a)), std::abs(out.stride(a))) <
        std::make_pair(std::abs(in.stride(b)), std::abs(out.stride(b)));
  });

  Batch b{in, out, {}, 1, in.stride(dim), out.stride(dim)};
  for (const auto d : dims) {
    b.numel *= in.size(d);
    const auto last = b.dims.empty() ? Dim{0, 0, 0} : b.dims.back();
    if (last.size && last.in_stride * last.size == in.stride(d) && last.out_stride * last.size == out.stride(d)) {
      b.dims.back().size *= in.size(d);
    } else {
      b.dims.push_back({in.size(d), in.stride(d), out.stride(d)});
    }
  }
  b.dims.resize(std::max<size_t>(b.dims.size(), 1), {1, 0, 0});
  return b;
}

// Per-pass options: four-step mapping and twiddles, Bluestein chirps (zero lengths = n), or one global stage
struct Pass {
  int64_t n_sig = 0;
  int64_t n_out = 0;
  int64_t ld_mul = 1;
  int64_t out_k_mul = 1;
  int64_t global = 0;
  int64_t ns = 1;
  double scale = 1;
  std::optional<Tensor> cl;
  std::optional<Tensor> cs;
  std::optional<Tensor> tt;
  Tensor tw; // W_n, or W_2n for packed real loads
  bool mirror_out = false;
  bool i0_dim1 = false;
};

// Encodes n-point transforms of the batch (half: packed real pairs)
void launch(id<MTLComputeCommandEncoder> enc, const Batch& b, int64_t n, bool inv, bool half, const Pass& x) {
  const auto radix = radices(x.global ? x.global : n, x.global ? kGlobalRadices : kRadices);
  const auto to_u32 = [](int64_t v) { return c10::checked_convert<uint32_t>(v, "uint32_t"); };
  const auto to_i32 = [](int64_t v) { return c10::checked_convert<int32_t>(v, "int32_t"); };
  const uint32_t n_u32 = to_u32(n);

  // Kernel family: a global radix, a fixed length, or the smallest radix set factoring n (p16: lengths 1 and 16)
  const auto set = n % 7 == 0 ? "p2357" : n % 3 == 0 || n % 5 == 0 ? "p235" : n == 1 || n == 16 ? "p16" : "p2";
  const auto has = [](const auto& list, int64_t v) { return std::ranges::find(list, v) != list.end(); };
  const auto family = x.global ? "g" + std::to_string(x.global) : has(kFixedSizes, n) ? std::to_string(n) : set;

  const auto n_sig = x.n_sig ? x.n_sig : n;
  const auto n_out = x.n_out ? x.n_out : n;
  const bool ld = x.cl || n_sig < n * x.ld_mul;
  const bool st = x.out_k_mul > 1 || x.cs;
  const bool tw4 = x.tt && !x.global;
  const uint32_t F = inv * kFFTInverse | half * kFFTPackedLoads | tw4 * kFFTFourStepTwiddles | st * kFFTExtendedStores |
      ld * kFFTExtendedLoads | (x.global > 0) * kFFTGlobalStage;

  const int64_t ept = fft_points_per_thread(n_u32, F);
  const int64_t rows = b.dims[0].size;
  // Threads per transform; global passes spread transforms over 64-thread groups, the unpack pass over 256
  const int64_t J = !x.global ? at::ceil_div(n, ept) : x.global == kFFTUnpackRadix ? 256 : 64;

  // Transforms per group, within smem: a strided axis tiles rows for 256 threads (4 to 32 rows), a contiguous
  // axis fills 64 threads and adds rows (up to 8) while more than 1 in 20 SIMD lanes would sit idle
  constexpr int64_t simd = c10::metal::simdgroup_size, max_threads = c10::metal::max_threadgroup_size;
  const bool strided = b.si != 1 || b.so != 1;
  const int64_t padded = n + (n >> kFFTPadShift) + 1; // Contiguous rows get a spare smem slot every 16
  int64_t B = strided ? std::min<int64_t>({kFFTMaxPoints / n, max_threads / J, std::max<int64_t>(4, 256 / J)}) : 64 / J;
  for (int64_t c = std::max<int64_t>(B, 1); !strided && c <= 8 && c * padded <= kFFTMaxPoints; c++) {
    if ((at::round_up(c * J, simd) - c * J) * 20 <= c * J) {
      B = c;
      break;
    }
  }
  B = std::clamp<int64_t>(B, 1, std::min<int64_t>(rows, 32));
  if (strided && rows > B && rows <= B + B / 2 && rows * n <= kFFTMaxPoints && rows * J <= max_threads) {
    B = rows; // A batch up to 1.5x the tile fits one group
  }
  const bool pad = !strided && B * (padded - (B == 1)) <= kFFTMaxPoints;
  const int64_t row = strided ? 1 : pad ? padded - (B == 1) : n + (B > 1);

  FFTParams params{
      .n = n_u32,
      .n_sig = to_u32(n_sig),
      .n_out = to_u32(n_out),
      .tgb = to_u32(B),
      .jt = to_u32(J),
      .ndim = to_u32(b.dims.size()),
      .nstages = to_u32(radix.size()),
      .in_stride = to_i32(b.si),
      .out_stride = to_i32(b.so),
      .scale = static_cast<float>(x.scale),
      .ld_mul = to_u32(x.ld_mul),
      .out_k_mul = to_u32(x.out_k_mul),
      .row = to_u32(row),
      .padshift = pad ? kFFTPadShift : kFFTNoPadShift,
      .es = to_u32(strided ? B : 1),
      .mirror_out = x.mirror_out,
      .i0_dim1 = x.i0_dim1,
      .via_smem = (b.si == 1) != (b.so == 1),
  };

  uint32_t ns = to_u32(x.ns);
  for (uint32_t s = 0; s < radix.size(); s++) {
    params.radix[s] = radix[s];
    params.ns[s] = ns;
    params.twstep[s] = n_u32 / radix[s] / ns * (half ? 2 : 1);
    params.inv_ns[s] = 1.0f / ns;
    ns *= radix[s];
  }

  for (const auto d : c10::irange(b.dims.size())) {
    params.sizes[d] = to_u32(b.dims[d].size);
    params.in_strides[d] = to_i32(b.dims[d].in_stride);
    params.out_strides[d] = to_i32(b.dims[d].out_stride);
  }

  const bool contig = !strided && B == 1 && has(kContigSizes, n) && !(F & kFFTMultiPass);
  const auto name = fmt::format(
      "fft_{}_{}_{}{}_{}", scalarToMetalTypeString(b.in), scalarToMetalTypeString(b.out), family, contig ? "c" : "", F);
  const int64_t ny = b.dims.size() > 1 ? b.dims[1].size : 1;
  const int64_t threads = x.global == kFFTUnpackRadix ? n / 2 + 1 : n / std::max<int64_t>(x.global, 1);
  const MTLSize grid = x.global ? MTLSizeMake(at::ceil_div(threads, J), rows, b.numel / rows)
                                : MTLSizeMake(at::ceil_div(rows, B), ny, b.numel / rows / ny);
  // Single contiguous rows get 1.5x their size (faster above 2048 points); global passes bind the 16-byte minimum
  const int64_t bytes = B * (strided ? n : row) * (strided || B > 1 ? 2 : 3) * sizeof(float);
  const int64_t smem = x.global ? 16 : std::min<int64_t>(kFFTSmemBytes, at::round_up<int64_t>(bytes, 16));
  const auto pso = lib.getPipelineStateForFunc(name);
  getMPSProfiler().beginProfileKernel(pso, "fft", {b.in, b.out}, getCurrentMPSStream());
  [enc setComputePipelineState:pso];
  mtl_setArgs(enc, b.in, b.out, x.cl, x.cs, x.tt, x.tw, params);
  [enc setThreadgroupMemoryLength:static_cast<NSUInteger>(smem) atIndex:0];
  [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(B * J, 1, 1)];
  getMPSProfiler().endProfileKernel(pso, getCurrentMPSStream());
}

// Encodes the launches of `encode` on the stream; tables and temps are made first, as ATen ops wait on its queue
template <class F>
void run(F encode) {
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      encode(stream->commandEncoder());
    }
  });
}

// Whether one threadgroup-memory pass handles n-point transforms of this batch
bool fits_one_pass(const Batch& b, int64_t n, bool chirped) {
  // Strided on both sides past 1024 points the tiles no longer coalesce: two tile passes move less
  const bool both = std::abs(b.si) != 1 && std::abs(b.so) != 1 && !chirped && n > kMaxCoalescedPoints;
  return n <= kFFTMaxPoints && !both;
}

// n-point transforms of the batch: x.n_out outputs
void fft_batch(const Batch& b, int64_t n, bool inv, const Pass& x) {
  const auto n_sig = x.n_sig ? x.n_sig : n;
  const auto n_out = x.n_out ? x.n_out : n;
  const bool real = !b.in.is_complex();
  const int64_t m = real && n % 2 == 0 ? n / 2 : n;
  const bool half = !x.cl && !x.cs && n > 2 && m < n && m <= kFFTMaxPoints;
  const int64_t pass_n = half ? m : n;
  if (fits_one_pass(b, pass_n, x.cs.has_value())) {
    Pass p = x;
    p.tw = table(Table::W, n);
    run([&](auto enc) { launch(enc, b, pass_n, inv, half, p); });
    return;
  }

  // Long non-pow2 lengths: one global-memory stage per radix between two contiguous temps, then the unpack
  const bool plain = !x.cl && !x.cs && !x.n_sig;
  const bool packs = !real || n % 2 == 0;
  auto gplan = radices(m, kGlobalRadices);
  if (n >= kLongPoints && (n & (n - 1)) && plain && packs && !gplan.empty()) {
    gplan.resize(gplan.size() + !b.in.is_complex(), kFFTUnpackRadix);
    const auto tmp = at::empty({2, b.numel * m}, b.out.options().dtype(kComplexFloat));
    const Tensor temps[] = {tmp[0], tmp[1]};
    const auto tt = table(Table::W2, m);
    const auto tw1 = table(Table::W, m), tw2 = real ? table(Table::W, 2 * m) : tw1;
    run([&](auto enc) {
      int64_t ns = 1;
      for (size_t s = 0; s < gplan.size(); s++) {
        const bool first = s == 0;
        const bool last = s + 1 == gplan.size();
        Batch bp{{}, {}, {}, b.numel, first ? b.si : 1, last ? b.so : 1};
        bp.in = first ? b.in : temps[s % 2];
        bp.out = last ? b.out : temps[(s + 1) % 2];
        int64_t tstride = m;
        for (int64_t d = 0; d < static_cast<int64_t>(b.dims.size()); d++) {
          bp.dims.push_back(
              {b.dims[d].size, first ? b.dims[d].in_stride : tstride, last ? b.dims[d].out_stride : tstride});
          tstride *= b.dims[d].size;
        }
        const bool packed = first && real;
        const Tensor& tw = packed ? tw2 : tw1;
        const Pass pass{
            .n_out = last ? n_out : m, .global = gplan[s], .ns = ns, .scale = last ? x.scale : 1, .tt = tt, .tw = tw};
        launch(enc, bp, m, inv, packed, pass);
        ns *= gplan[s];
      }
    });
    return;
  }

  // Four-step: n = n1*n2, temp [outer batch][n2][n1][inner batch]; pass 1 transforms strided n1-point columns
  // times W_n^(k1*i0), pass 2 writes k1+n1*k2 (n1 <= 256 if it recurses). Real transforms halve one pass.
  const bool big = plain && b.numel * n >= kLongPoints;
  bool r2c = big && !b.in.is_complex();
  const bool nest = !x.cs && n_out == n;

  // n1: the largest smooth divisor <= sqrt(n) whose cofactor fits a pass; nr: with an even halved real pass
  int64_t n1 = 0;
  int64_t nr = 0;
  for (int64_t d = 1; d * d <= n; d++) {
    const int64_t q = n / d;
    if (n % d || !is_smooth(d) || !is_smooth(q)) {
      continue;
    }
    nr = q <= kFFTMaxPoints && d % 2 == 0 && d > 2 && d <= 2 * kFFTMaxPoints ? d : nr;
    n1 = (q <= kFFTMaxPoints || nest) && (!nest || d <= 256) ? d : n1;
  }
  r2c = r2c && nr;
  n1 = r2c ? nr : n1;
  TORCH_CHECK_NOT_IMPLEMENTED(n1 > 0, "MPS FFT does not support transforms of length ", n);

  const auto n2 = n / n1;
  const auto rl = r2c ? n1 / 2 + 1 : n1;
  const bool inner = b.dims[0].size > 1 && std::abs(b.dims[0].in_stride) < std::abs(b.si);
  const int64_t ninner = inner ? b.dims[0].size : 1;
  Batch b1{b.in, {}, {}, b.numel * n2, b.si * n2, ninner};
  Batch b2{{}, b.out, {}, b.numel * rl, rl * ninner, b.so * n1};
  int64_t t = 1;
  for (int64_t d = 0; d < static_cast<int64_t>(b.dims.size()); d++) {
    const int64_t ts = d < inner ? t : t * n2 * rl; // The temp's stride of batch dim d
    b1.dims.push_back({b.dims[d].size, b.dims[d].in_stride, ts});
    b2.dims.push_back({b.dims[d].size, ts, b.dims[d].out_stride});
    t *= b.dims[d].size;
  }
  b1.dims.insert(b1.dims.begin() + inner, Dim{n2, b.si, rl * ninner});
  b2.dims.insert(b2.dims.begin() + inner, Dim{rl, ninner, b.so});

  const bool inplace = !r2c && n_out == n && b.so == ninner && !b.in.is_alias_of(b.out) &&
      std::all_of(
          b2.dims.begin(), b2.dims.end(), [](const Dim& d) { return d.size == 1 || d.in_stride == d.out_stride; });
  const auto tmp = inplace ? b.out : at::empty({b.numel * n2 * rl}, b.out.options().dtype(kComplexFloat));
  b1.out = b2.in = tmp;
  const bool two = r2c || n2 <= kMaxCoalescedPoints || !nest;
  const auto tt = table(Table::W2, n), tw1 = table(Table::W, n1), tw2 = two ? table(Table::W, n2) : Tensor();
  const Pass load{.n_sig = n_sig, .n_out = rl, .ld_mul = n2, .cl = x.cl, .tt = tt, .tw = tw1, .i0_dim1 = inner};
  if (two) {
    const Pass store{.n_sig = n,
                     .n_out = n_out,
                     .out_k_mul = n1,
                     .scale = x.scale,
                     .cs = x.cs,
                     .tw = tw2,
                     .mirror_out = r2c,
                     .i0_dim1 = inner};
    run([&](auto enc) {
      launch(enc, b1, r2c ? n1 / 2 : n1, inv, r2c, load);
      launch(enc, b2, n2, inv, false, store);
    });
    return;
  }

  run([&](auto enc) { launch(enc, b1, n1, inv, false, load); });
  fft_batch(b2, n2, inv, {.scale = x.scale});
}

// n-point transforms along `dim`; lengths with other prime factors use Bluestein's algorithm
void fft_pass(const Tensor& in, const Tensor& out, int64_t dim, int64_t n, bool inv, const Pass& x = {}) {
  dim = maybe_wrap_dim(dim, in.dim());
  if (!canUse32BitIndexMath(in) || !canUse32BitIndexMath(out)) {
    // Kernels index with 32 bits: halve the largest batch dim
    int64_t d = dim == 0;
    for (const auto i : c10::irange(in.dim())) {
      d = i != dim && in.size(i) > in.size(d) ? i : d;
    }
    TORCH_CHECK_NOT_IMPLEMENTED(in.dim() > 1 && in.size(d) > 1,
                                "MPS FFT: a transform must have fewer than 2^31 elements");
    const auto h = in.size(d) / 2;
    fft_pass(in.narrow(d, 0, h), out.narrow(d, 0, h), dim, n, inv, x);
    return fft_pass(in.narrow(d, h, in.size(d) - h), out.narrow(d, h, in.size(d) - h), dim, n, inv, x);
  }
  if (is_smooth(n)) {
    return fft_batch(collapse(in, out, dim), n, inv, x);
  }

  // Bluestein: X = c * IFFT_M(FFT_M(x * c) * B)[:n], M a power of two >= 2n - 1 (at most two tile passes, 2^24),
  // chirps folded into the kernels
  TORCH_CHECK_NOT_IMPLEMENTED(n <= 1 << 23, "MPS FFT: lengths with prime factors above 7 are limited to 2^23");
  const auto c = table(Table::Chirp, n, inv), B = table(Table::ChirpFFT, n, inv);
  const auto M = B.numel();
  DimVector sizes(in.sizes());
  sizes[dim] = M;
  auto tmp = at::empty(sizes, in.options().dtype(kComplexFloat)); // Several passes always run in fp32
  fft_pass(in, tmp, dim, M, false, {.n_sig = n, .scale = x.scale / M, .cl = c, .cs = B});
  fft_pass(tmp, out, dim, M, true, {.n_out = x.n_out ? x.n_out : n, .cs = c});
}

// fp16 partials overflow and lose precision between passes, so transforms taking more than one pass run in fp32
bool needs_fp32(const Tensor& in, const Tensor& out, IntArrayRef dim, int64_t n) {
  const bool halved = !in.is_complex() && n % 2 == 0 && n > 2;
  return c10::toRealValueType(in.scalar_type()) == kHalf &&
      (dim.size() > 1 || !is_smooth(n) ||
       !fits_one_pass(collapse(in, out, maybe_wrap_dim(dim.back(), in.dim())), halved ? n / 2 : n, false));
}

// Real lengths beyond two tile passes (powers of two and odd lengths above 2^24) run as complex transforms
bool beyond_real_passes(int64_t n) {
  return n > kFFTMaxPoints * kFFTMaxPoints && (!(n & (n - 1)) || n % 2);
}

double norm_scale(int64_t normalization, const Tensor& t, IntArrayRef dims) {
  const auto n = std::accumulate(dims.begin(), dims.end(), 1.0, [&](double a, int64_t d) { return a * t.size(d); });
  return std::pow(n, -0.5 * normalization); // fft_norm_mode: none, by_root_n, by_n
}

} // anonymous namespace

Tensor _fft_c2r_mps(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  auto out = at::empty({0}, self.options().dtype(c10::toRealValueType(self.scalar_type())));
  return _fft_c2r_mps_out(self, dim, normalization, last_dim_size, out);
}

Tensor _fft_r2c_mps(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  auto out = at::empty({0}, self.options().dtype(c10::toComplexType(self.scalar_type())));
  return _fft_r2c_mps_out(self, dim, normalization, onesided, out);
}

Tensor _fft_c2c_mps(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  TORCH_CHECK(self.is_complex());
  if (dim.empty()) {
    return self.clone();
  }
  auto out = at::empty(self.sizes(), self.options());
  return _fft_c2c_mps_out(self, dim, normalization, forward, out);
}

Tensor& _fft_r2c_mps_out(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided, Tensor& out) {
  TORCH_CHECK(self.scalar_type() == kFloat || self.scalar_type() == kHalf, "Only float and half dtypes are supported");
  TORCH_CHECK(out.scalar_type() == c10::toComplexType(self.scalar_type()));
  TORCH_CHECK(out.device() == self.device(), "Expected out tensor on ", self.device(), " but got ", out.device());
  const auto input_sizes = self.sym_sizes();
  SymDimVector out_sizes(input_sizes.begin(), input_sizes.end());
  const auto last_dim = maybe_wrap_dim(dim.back(), self.dim());
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }
  at::native::resize_output_symint(out, out_sizes);
  if (self.numel() == 0) {
    return out.zero_();
  }

  const auto n = self.size(last_dim);
  if (needs_fp32(self, out, dim, n)) {
    return out.copy_(_fft_r2c_mps(self.to(kFloat), dim, normalization, onesided));
  }
  if (beyond_real_passes(n)) {
    const auto full = at::_fft_c2c(self.to(out.scalar_type()), dim, normalization, true);
    return out.copy_(full.narrow(last_dim, 0, out.size(last_dim)));
  }
  const Pass x{.n_out = out.size(last_dim), .scale = norm_scale(normalization, self, dim)};
  fft_pass(self, out, last_dim, n, false, x);
  for (const auto i : c10::irange(dim.size() - 1)) {
    fft_pass(out, out, dim[i], self.size(dim[i]), false);
  }
  return out;
}

Tensor& _fft_c2r_mps_out(const Tensor& self,
                         IntArrayRef dim,
                         int64_t normalization,
                         int64_t last_dim_size,
                         Tensor& out) {
  TORCH_CHECK(self.is_complex(), "Input must be complex");
  TORCH_CHECK(out.scalar_type() == c10::toRealValueType(self.scalar_type()), "Unexpected output type");
  TORCH_CHECK(out.device() == self.device(), "Expected out tensor on ", self.device(), " but got ", out.device());
  // All public callers resize the input to last_dim_size / 2 + 1; any other size aborts in HermiteanToRealFFT
  TORCH_INTERNAL_ASSERT(self.size(dim.back()) == last_dim_size / 2 + 1,
                        "_fft_c2r: expected ",
                        last_dim_size / 2 + 1,
                        " elements in dim ",
                        dim.back(),
                        ", but got ",
                        self.size(dim.back()));
  const auto in_sizes = self.sym_sizes();
  SymDimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = last_dim_size;
  at::native::resize_output_symint(out, out_sizes);
  if (out.numel() == 0) {
    return out;
  }

  auto key = __func__ + getTensorsStringKey({self}) + ":" + getArrayRefString(dim) + ":" +
      std::to_string(normalization) + ":" + std::to_string(last_dim_size);
  @autoreleasepool {
    auto cachedGraph = LookUpOrCreateCachedGraph<MPSUnaryCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      auto axisPlan = computeFFTAxisPlan(dim, self.dim());
      auto fftInput = applyFFTInputPlan(mpsGraph, inputTensor, axisPlan);
      auto descriptor = [MPSGraphFFTDescriptor descriptor];
      descriptor.scalingMode = normalization_to_ScalingMode(normalization);
      descriptor.inverse = YES;
      descriptor.roundToOddHermitean = ((last_dim_size % 2) == 1) ? YES : NO;
      auto outputTensor = [mpsGraph HermiteanToRealFFTWithTensor:fftInput
                                                            axes:axisPlan.axes
                                                      descriptor:descriptor
                                                            name:nil];
      outputTensor = applyFFTOutputPlan(mpsGraph, outputTensor, axisPlan);
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });
    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return out;
}

Tensor& _fft_c2c_mps_out(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward, Tensor& out) {
  TORCH_CHECK(self.is_complex());
  TORCH_CHECK(out.device() == self.device(), "Expected out tensor on ", self.device(), " but got ", out.device());
  at::native::resize_output_symint(out, self.sym_sizes());
  if (out.numel() == 0) {
    return out;
  }

  if (dim.empty()) {
    return out.copy_(self);
  }
  if (needs_fp32(self, out, dim, self.size(dim.back()))) {
    return out.copy_(_fft_c2c_mps(self.to(kComplexFloat), dim, normalization, forward));
  }
  const auto scale = norm_scale(normalization, self, dim);
  for (const auto i : c10::irange(dim.size())) {
    fft_pass(i ? out : self, out, dim[i], self.size(dim[i]), !forward, {.scale = i ? 1.0 : scale});
  }
  return out;
}

} // namespace at::native
