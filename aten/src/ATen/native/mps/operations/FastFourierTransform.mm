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
#include <ATen/ops/_fft_c2c_native.h>
#include <ATen/ops/_fft_c2r_native.h>
#include <ATen/ops/_fft_r2c_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/mul.h>
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
using namespace std::literals;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/FFT_metallib.h>
#endif

// Radix sequence of n over `rs` (largest first; {1} for n = 1), empty when n has other prime factors
c10::SmallVector<uint32_t, kFFTMaxStages> radices(int64_t n, c10::ArrayRef<int> rs = kFFTRadices) {
  c10::SmallVector<uint32_t, kFFTMaxStages> plan(n == 1, 1);
  for (const auto r : rs) {
    for (; n % r == 0; n /= r) {
      plan.push_back(r);
    }
  }
  return n == 1 ? plan : decltype(plan){};
}

bool is_smooth(int64_t n) {
  return !radices(n).empty();
}

// Bluestein's convolution length for n points: a power of two >= 2n - 1
int64_t chirp_fft_size(int64_t n) {
  return static_cast<int64_t>(std::bit_ceil<uint64_t>(2 * n - 1));
}

// Cached MPS tables. W: W_n^k for k < n. W2, for long axes: W_n^b for b < 4096 then W_n^(4096 a). Chirp: Bluestein's
// exp(-+ i pi k^2 / n), and ChirpFFT: the chirp_fft_size(n)-point FFT of its conjugate wrapped around zero
enum class Table { W, W2, Chirp, ChirpFFT };

Tensor table(Table kind, int64_t n, bool inverse = false) {
  static std::mutex mutex;
  static std::map<std::tuple<Table, int64_t, bool>, Tensor> cache;
  std::lock_guard<std::mutex> guard(mutex);
  auto& t = cache[{kind, n, inverse}];
  if (t.defined()) {
    return t;
  }
  const auto k = kind == Table::W2
      ? at::cat({at::arange(kFFTTwiddleBlock, at::kDouble), at::arange(0, n, kFFTTwiddleBlock, at::kDouble)})
      : at::arange(n, at::kDouble);
  const auto angle = kind == Table::W || kind == Table::W2
      ? k.mul(-2 * M_PI / n)
      : k.mul(k).remainder(2 * n).mul((inverse ? 1 : -1) * M_PI / n);
  const auto c = at::polar(at::ones_like(angle), angle);
  const auto M = chirp_fft_size(n);
  const auto b = kind == Table::ChirpFFT
      ? at::fft_fft(at::cat({c.conj(), at::zeros({M - 2 * n + 1}, c.options()), c.narrow(0, 1, n - 1).conj().flip(0)}))
      : c;
  return t = b.to(at::device(kMPS).dtype(kComplexFloat));
}

struct Dim {
  int64_t size, in_stride, out_stride;
};

struct Batch {
  Tensor in;
  Tensor out;
  c10::SmallVector<Dim, c10::metal::max_ndim> dims; // Innermost first
  int64_t numel, si, so; // Transforms, and the element strides along the axis
  bool wide = false; // Offsets need 64 bits
};

// Batch dims of a pass along `dim` (every other dim), innermost first, merged where both tensors allow it
Batch collapse(const Tensor& in, const Tensor& out, int64_t dim) {
  c10::SmallVector<int64_t, c10::metal::max_ndim> dims;
  for (const auto d : c10::irange(in.dim())) {
    if (d != dim && in.size(d) != 1) {
      dims.push_back(d);
    }
  }
  std::ranges::sort(dims, {}, [&](auto d) { return std::make_pair(std::abs(in.stride(d)), std::abs(out.stride(d))); });

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
  Tensor tw; // W_n
  bool i0_dim1 = false;
};

// Encodes n-point transforms of the batch with the kernels whose offsets are of type index_t
template <typename index_t>
void launch_as(id<MTLComputeCommandEncoder> enc, const Batch& b, int64_t n, bool inv, const Pass& x) {
  const auto radix = x.global ? radices(x.global, kFFTGlobalRadices) : radices(n);
  const auto to_u32 = [](int64_t v) { return c10::checked_convert<uint32_t>(v, "uint32_t"); };
  const auto to_off = [](int64_t v) { return c10::checked_convert<index_t>(v, "offset"); };
  const uint32_t n_u32 = to_u32(n);

  // Kernel family: a global radix, a fixed length, or the smallest radix set factoring n (p16: lengths 1 and 16);
  // the names match the FAMILY instantiations in FFT.metal
  const auto set = n % 7 == 0 ? "p2357" : n % 3 == 0 || n % 5 == 0 ? "p235" : n == 1 || n == 16 ? "p16" : "p2";
  const auto has = [](const auto& list, int64_t v) { return std::ranges::find(list, v) != std::end(list); };
  const auto family = x.global ? "g" + std::to_string(x.global) : has(kFFTFixedSizes, n) ? std::to_string(n) : set;

  const auto n_sig = x.n_sig ? x.n_sig : n;
  const auto n_out = x.n_out ? x.n_out : n;
  const bool ld = x.cl || n_sig < n * x.ld_mul;
  const bool st = x.out_k_mul > 1 || x.cs;
  const bool tw4 = x.tt && !x.global;
  const uint32_t F = inv * kFFTInverse | tw4 * kFFTFourStepTwiddles | st * kFFTExtendedStores | ld * kFFTExtendedLoads |
      (x.global > 0) * kFFTGlobalStage;

  const int64_t ept = fft_points_per_thread(n_u32, F);
  const int64_t rows = b.dims[0].size;
  // Threads per transform
  const int64_t J = !x.global ? at::ceil_div(n, ept) : kFFTGlobalThreads;

  // Transforms per group, within smem
  const bool strided = b.si != 1 || b.so != 1;
  const int64_t padded = n + (n >> kFFTPadShift) + 1; // Contiguous rows get a spare smem slot every 16
  int64_t B = strided
      ? std::min<int64_t>(
            {kFFTMaxPoints / n, kFFTMaxThreads / J, std::max<int64_t>(kFFTMinStridedRows, kFFTStridedThreads / J)})
      : kFFTContiguousThreads / J;
  for (int64_t c = std::max<int64_t>(B, 1); !strided && c <= kFFTMaxContiguousRows && c * padded <= kFFTMaxPoints;
       c++) {
    if ((at::round_up<int64_t>(c * J, c10::metal::simdgroup_size) - c * J) * kFFTIdleLaneDivisor <= c * J) {
      B = c;
      break;
    }
  }
  B = std::clamp<int64_t>(B, 1, std::min(rows, kFFTMaxRows));
  if (strided && rows > B && rows <= B + B / 2 && rows * n <= kFFTMaxPoints && rows * J <= kFFTMaxThreads) {
    B = rows; // A batch up to 1.5x the tile fits one group
  }
  const bool pad = !strided && B * (padded - (B == 1)) <= kFFTMaxPoints;
  const int64_t row = strided ? 1 : pad ? padded - (B == 1) : n + (B > 1);

  FFTParams<index_t> params{
      .n = n_u32,
      .n_sig = to_u32(n_sig),
      .n_out = to_u32(n_out),
      .tgb = to_u32(B),
      .jt = to_u32(J),
      .ndim = to_u32(b.dims.size()),
      .nstages = to_u32(radix.size()),
      .in_stride = to_off(b.si),
      .out_stride = to_off(b.so),
      .scale = static_cast<float>(x.scale),
      .ld_mul = to_u32(x.ld_mul),
      .out_k_mul = to_u32(x.out_k_mul),
      .row = to_u32(row),
      .padshift = pad ? kFFTPadShift : kFFTNoPadShift,
      .es = to_u32(strided ? B : 1),
      .i0_dim1 = x.i0_dim1,
      .via_smem = (b.si == 1) != (b.so == 1),
  };

  uint32_t ns = to_u32(x.ns);
  for (uint32_t s = 0; s < radix.size(); s++) {
    params.radix[s] = radix[s];
    params.ns[s] = ns;
    params.twstep[s] = n_u32 / radix[s] / ns;
    params.inv_ns[s] = 1.0f / ns;
    ns *= radix[s];
  }

  for (const auto d : c10::irange(b.dims.size())) {
    params.sizes[d] = to_u32(b.dims[d].size);
    params.in_strides[d] = to_off(b.dims[d].in_stride);
    params.out_strides[d] = to_off(b.dims[d].out_stride);
  }

  const bool contig = !strided && B == 1 && has(kFFTContigSizes, n) && !(F & kFFTMultiPass);
  const auto suffix = contig ? "c"sv : b.wide ? "w"sv : ""sv;
  const auto name = fmt::format(
      "fft_{}_{}_{}{}_{}", scalarToMetalTypeString(b.in), scalarToMetalTypeString(b.out), family, suffix, F);
  const int64_t ny = b.dims.size() > 1 ? b.dims[1].size : 1;
  const int64_t threads = n / std::max<int64_t>(x.global, 1);
  const MTLSize grid = x.global ? MTLSizeMake(at::ceil_div(threads, J), rows, b.numel / rows)
                                : MTLSizeMake(at::ceil_div(rows, B), ny, b.numel / rows / ny);
  // Single contiguous rows get 1.5x their size (faster above 2048 points)
  const int64_t bytes = B * (strided ? n : row) * (strided || B > 1 ? 2 : 3) * sizeof(float);
  const int64_t smem =
      x.global ? kFFTMinSmemBytes : std::min<int64_t>(kFFTSmemBytes, at::round_up(bytes, kFFTMinSmemBytes));
  const auto pso = lib.getPipelineStateForFunc(name);
  TORCH_INTERNAL_ASSERT(B * J <= static_cast<int64_t>([pso maxTotalThreadsPerThreadgroup]));
  getMPSProfiler().beginProfileKernel(pso, "fft", {b.in, b.out}, getCurrentMPSStream());
  [enc setComputePipelineState:pso];
  mtl_setArgs(enc, b.in, b.out, x.cl, x.cs, x.tt, x.tw, params);
  [enc setThreadgroupMemoryLength:static_cast<NSUInteger>(smem) atIndex:0];
  [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(B * J, 1, 1)];
  getMPSProfiler().endProfileKernel(pso, getCurrentMPSStream());
}

void launch(id<MTLComputeCommandEncoder> enc, const Batch& b, int64_t n, bool inv, const Pass& x) {
  return b.wide ? launch_as<int64_t>(enc, b, n, inv, x) : launch_as<int32_t>(enc, b, n, inv, x);
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
  const bool both = std::abs(b.si) != 1 && std::abs(b.so) != 1 && !chirped && n > kFFTMaxCoalescedPoints;
  return n <= kFFTMaxPoints && !both;
}

// n-point transforms of the batch: x.n_out outputs
void fft_batch(const Batch& b, int64_t n, bool inv, const Pass& x) {
  const auto n_sig = x.n_sig ? x.n_sig : n;
  const auto n_out = x.n_out ? x.n_out : n;
  if (fits_one_pass(b, n, x.cs.has_value())) {
    Pass p = x;
    p.tw = table(Table::W, n);
    run([&](auto enc) { launch(enc, b, n, inv, p); });
    return;
  }

  // Long non-pow2 lengths: one global-memory stage per radix between two contiguous temps
  const bool plain = !x.cl && !x.cs && !x.n_sig;
  const auto gplan = radices(n, kFFTGlobalRadices);
  if (n >= kFFTLongPoints && (n & (n - 1)) && plain && !b.wide && !gplan.empty()) {
    const auto tmp = at::empty({2, b.numel * n}, b.out.options().dtype(kComplexFloat));
    const Tensor temps[] = {tmp[0], tmp[1]};
    const auto tt = table(Table::W2, n), tw = table(Table::W, n);
    run([&](auto enc) {
      int64_t ns = 1;
      for (size_t s = 0; s < gplan.size(); s++) {
        const bool first = s == 0;
        const bool last = s + 1 == gplan.size();
        Batch bp{{}, {}, {}, b.numel, first ? b.si : 1, last ? b.so : 1};
        bp.in = first ? b.in : temps[s % 2];
        bp.out = last ? b.out : temps[(s + 1) % 2];
        int64_t tstride = n;
        for (int64_t d = 0; d < static_cast<int64_t>(b.dims.size()); d++) {
          bp.dims.push_back(
              {b.dims[d].size, first ? b.dims[d].in_stride : tstride, last ? b.dims[d].out_stride : tstride});
          tstride *= b.dims[d].size;
        }
        const Pass pass{
            .n_out = last ? n_out : n, .global = gplan[s], .ns = ns, .scale = last ? x.scale : 1, .tt = tt, .tw = tw};
        launch(enc, bp, n, inv, pass);
        ns *= gplan[s];
      }
    });
    return;
  }

  // Four-step: n = n1*n2, temp [outer batch][n2][n1][inner batch]; pass 1 transforms strided n1-point columns
  // times W_n^(k1*i0), pass 2 writes k1+n1*k2 (n1 <= kFFTMaxNestedColumns if it recurses).
  const bool nest = !x.cs && n_out == n;

  // n1: the largest smooth divisor <= sqrt(n) whose cofactor fits a pass
  int64_t n1 = 0;
  for (int64_t d = 1; d * d <= n; d++) {
    const int64_t q = n / d;
    if (n % d || !is_smooth(d) || !is_smooth(q)) {
      continue;
    }
    n1 = (q <= kFFTMaxPoints || nest) && (!nest || d <= kFFTMaxNestedColumns) ? d : n1;
  }
  TORCH_CHECK_NOT_IMPLEMENTED(n1 > 0, "MPS FFT does not support transforms of length ", n);

  const auto n2 = n / n1;
  const bool inner = b.dims[0].size > 1 && std::abs(b.dims[0].in_stride) < std::abs(b.si);
  const int64_t ninner = inner ? b.dims[0].size : 1;
  Batch b1{b.in, {}, {}, b.numel * n2, b.si * n2, ninner};
  Batch b2{{}, b.out, {}, b.numel * n1, n1 * ninner, b.so * n1};
  b1.wide = b2.wide = b.wide;
  int64_t t = 1;
  for (int64_t d = 0; d < static_cast<int64_t>(b.dims.size()); d++) {
    const int64_t ts = d < inner ? t : t * n; // The temp's stride of batch dim d
    b1.dims.push_back({b.dims[d].size, b.dims[d].in_stride, ts});
    b2.dims.push_back({b.dims[d].size, ts, b.dims[d].out_stride});
    t *= b.dims[d].size;
  }
  b1.dims.insert(b1.dims.begin() + inner, Dim{n2, b.si, n1 * ninner});
  b2.dims.insert(b2.dims.begin() + inner, Dim{n1, ninner, b.so});

  const bool inplace = n_out == n && b.so == ninner && !b.in.is_alias_of(b.out) &&
      std::all_of(b2.dims.begin(), b2.dims.end(), [](const Dim& d) {
                         return d.size == 1 || d.in_stride == d.out_stride;
                       });
  const auto tmp = inplace ? b.out : at::empty({b.numel * n}, b.out.options().dtype(kComplexFloat));
  b1.out = b2.in = tmp;
  const bool two = n2 <= kFFTMaxCoalescedPoints || !nest;
  const auto tt = table(Table::W2, n), tw1 = table(Table::W, n1), tw2 = two ? table(Table::W, n2) : Tensor();
  const Pass load{.n_sig = n_sig, .n_out = n1, .ld_mul = n2, .cl = x.cl, .tt = tt, .tw = tw1, .i0_dim1 = inner};
  if (two) {
    const Pass store{
        .n_sig = n, .n_out = n_out, .out_k_mul = n1, .scale = x.scale, .cs = x.cs, .tw = tw2, .i0_dim1 = inner};
    run([&](auto enc) {
      launch(enc, b1, n1, inv, load);
      launch(enc, b2, n2, inv, store);
    });
    return;
  }

  run([&](auto enc) { launch(enc, b1, n1, inv, load); });
  fft_batch(b2, n2, inv, {.scale = x.scale});
}

// n-point transforms along `dim`; lengths with other prime factors use Bluestein's algorithm
void fft_pass(const Tensor& in, const Tensor& out, int64_t dim, int64_t n, bool inv, const Pass& x = {}) {
  dim = maybe_wrap_dim(dim, in.dim());
  constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
  // One transform spanning 2^31 elements takes the kernels with 64-bit offsets
  const auto reach = [&](const Tensor& t) { return (t.size(dim) - 1) * t.stride(dim); };
  const bool wide = n < kMax && std::max(reach(in), reach(out)) >= kMax;
  const auto points = std::min<int64_t>(n, kFFTMaxPoints);
  const auto threads = out.numel() / out.size(dim) * at::ceil_div<int64_t>(points, fft_points_per_thread(points, 0));
  if (wide ? threads > kMax : !canUse32BitIndexMath(in) || !canUse32BitIndexMath(out)) {
    // Halve the batch dim spanning the most elements
    const auto span = [&](int64_t i) { return (in.size(i) - 1) * std::max(in.stride(i), out.stride(i)); };
    int64_t d = -1;
    for (const auto i : c10::irange(in.dim())) {
      d = i != dim && in.size(i) > 1 && (d < 0 || span(i) > span(d)) ? i : d;
    }
    TORCH_CHECK_NOT_IMPLEMENTED(d >= 0, "MPS FFT: a transform must have fewer than 2^31 elements");
    const auto h = in.size(d) / 2;
    fft_pass(in.narrow(d, 0, h), out.narrow(d, 0, h), dim, n, inv, x);
    return fft_pass(in.narrow(d, h, in.size(d) - h), out.narrow(d, h, in.size(d) - h), dim, n, inv, x);
  }
  if (is_smooth(n)) {
    auto b = collapse(in, out, dim);
    b.wide = wide;
    return fft_batch(b, n, inv, x);
  }

  // Bluestein: X = c * IFFT_M(FFT_M(x * c) * B)[:n], M = chirp_fft_size(n)
  TORCH_CHECK_NOT_IMPLEMENTED(!wide, "MPS FFT: lengths with prime factors above 7 cannot span 2^31 elements");
  const auto M = chirp_fft_size(n);
  TORCH_CHECK_NOT_IMPLEMENTED(M < kMax, "MPS FFT: lengths with prime factors above 7 are limited to 2^29");
  const auto c = table(Table::Chirp, n, inv), B = table(Table::ChirpFFT, n, inv);
  // The temp's axis goes last where its transforms would otherwise span 2^31 elements
  const bool last = (M - 1) * c10::multiply_integers(in.sizes().slice(dim + 1)) >= kMax;
  DimVector sizes(in.sizes());
  sizes[dim] = M;
  std::rotate(sizes.begin() + dim, sizes.begin() + dim + last, sizes.end());
  auto tmp = at::empty(sizes, in.options().dtype(kComplexFloat)); // Several passes always run in fp32
  tmp = last ? tmp.movedim(-1, dim) : std::move(tmp);
  // Two tile passes fold every chirp into the kernels
  if (M <= kFFTMaxPoints * kFFTMaxPoints) {
    fft_pass(in, tmp, dim, M, false, {.n_sig = n, .scale = x.scale / M, .cl = c, .cs = B});
    return fft_pass(tmp, out, dim, M, true, {.n_out = n, .cs = c});
  }
  // Longer transforms nest and their stores fold none: B goes into the second transform's loads, c into a mul
  fft_pass(in, tmp, dim, M, false, {.n_sig = n, .scale = x.scale / M, .cl = c});
  fft_pass(tmp, tmp, dim, M, true, {.cl = B});
  auto dst = out.movedim(dim, -1); // c broadcasts along the last dim
  at::mul_out(dst, tmp.narrow(dim, 0, n).movedim(dim, -1), c);
}

// fp16 partials overflow and lose precision between passes, so transforms taking more than one pass run in fp32
bool needs_fp32(const Tensor& in, const Tensor& out, IntArrayRef dim, int64_t n) {
  return c10::toRealValueType(in.scalar_type()) == kHalf &&
      (dim.size() > 1 || !is_smooth(n) ||
       !fits_one_pass(collapse(in, out, maybe_wrap_dim(dim.back(), in.dim())), n, false));
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
  auto out = at::empty(self.sizes(), self.options());
  return _fft_c2c_mps_out(self, dim, normalization, forward, out);
}

// TODO: Investigate numerical discrepancies see https://github.com/pytorch/pytorch/issues/120237
Tensor& _fft_r2c_mps_out(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided, Tensor& out) {
  TORCH_CHECK(self.scalar_type() == kFloat || self.scalar_type() == kHalf, "Only float and half dtypes are supported");
  TORCH_CHECK(out.scalar_type() == c10::toComplexType(self.scalar_type()));
  TORCH_CHECK(out.device() == self.device(), "Expected out tensor on ", self.device(), " but got ", out.device());
  const auto input_sizes = self.sym_sizes();
  SymDimVector out_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }
  at::native::resize_output_symint(out, out_sizes);
  if (out.numel() == 0) {
    return out;
  }

  auto key = __func__ + getTensorsStringKey({self, out}) + ":" + getArrayRefString(dim) + ":" +
      std::to_string(normalization) + ":" + std::to_string(onesided);
  @autoreleasepool {
    auto cachedGraph = LookUpOrCreateCachedGraph<MPSUnaryCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      auto axisPlan = computeFFTAxisPlan(dim, self.dim());
      auto fftInput = applyFFTInputPlan(mpsGraph, inputTensor, axisPlan);
      auto descriptor = [MPSGraphFFTDescriptor descriptor];
      descriptor.scalingMode = normalization_to_ScalingMode(normalization);
      MPSGraphTensor* outputTensor;
      if (onesided) {
        // Return only unique results:
        outputTensor = [mpsGraph realToHermiteanFFTWithTensor:fftInput
                                                         axes:axisPlan.axes
                                                   descriptor:descriptor
                                                         name:nil];
      } else {
        // Return with Hermitean conjugate results:
        auto useDataType =
            (fftInput.dataType == MPSDataTypeFloat16) ? MPSDataTypeComplexFloat16 : MPSDataTypeComplexFloat32;
        auto cTensor = [mpsGraph castTensor:fftInput toType:useDataType name:nil];
        outputTensor = [mpsGraph fastFourierTransformWithTensor:cTensor
                                                           axes:axisPlan.axes
                                                     descriptor:descriptor
                                                           name:nil];
      }
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
