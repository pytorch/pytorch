#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/AdaptivePooling.h>
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/Pool.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <type_traits>
#include <ATen/OpMathType.h>
#include <ATen/native/ReduceOpsUtils.h>

namespace at::native {

namespace {

template <typename scalar_t>
bool is_nan(scalar_t v) {
  if (std::is_integral_v<scalar_t> || std::is_same_v<scalar_t, unsigned char>) {
    return false;
  }
  return std::isnan(v);
}

template <typename scalar_t>
vec::Vectorized<scalar_t> is_nan_vec(vec::Vectorized<scalar_t> vec) {
  return vec.isnan();
}

// TODO: use is_integeral/is_same to check the scalar_t and simplify the implementation
// currently it does not work
template <>
vec::Vectorized<unsigned char> is_nan_vec<unsigned char>(vec::Vectorized<unsigned char> vec) {
  Vectorized<unsigned char> ret(false);
  return ret;
}

template <>
vec::Vectorized<signed char> is_nan_vec<signed char>(vec::Vectorized<signed char> vec) {
  Vectorized<signed char> ret(false);
  return ret;
}

template <>
vec::Vectorized<short> is_nan_vec<short>(vec::Vectorized<short> vec) {
  Vectorized<short> ret(false);
  return ret;
}

template <>
vec::Vectorized<int> is_nan_vec<int>(vec::Vectorized<int> vec) {
  Vectorized<int> ret(false);
  return ret;
}

template <>
vec::Vectorized<int64_t> is_nan_vec<int64_t>(vec::Vectorized<int64_t> vec) {
  Vectorized<int64_t> ret(false);
  return ret;
}

template <typename scalar_t, typename opmath_t>
inline
std::enable_if_t<std::is_same_v<scalar_t, opmath_t>, void>
compute_internal(
  const scalar_t* input_data,
  scalar_t* out_data,
  opmath_t* max_ptr,
  vec::int_same_size_t<opmath_t>* index_ptr,
  int64_t* ind,
  int64_t input_depth, int64_t input_height, int64_t input_width, int64_t channels,
  int64_t n,
  int64_t len,
  int64_t size,
  int64_t id0, int64_t id1,
  int64_t ih0, int64_t ih1,
  int64_t iw0, int64_t iw1,
  int64_t dilationD,
  int64_t dilationH,
  int64_t dilationW) {
  using Vec = vec::Vectorized<scalar_t>;
  using integer_t = vec::int_same_size_t<opmath_t>;
  using iVec = vec::Vectorized<integer_t>;
  // Pass I: init out lane
  iVec index0_vec = iVec(id0 * input_height * input_width + ih0 * input_width + iw0);

  scalar_t min_value = lower_bound<scalar_t>();
  Vec out_vec = Vec(min_value);
  int64_t d1 = 0;
  for (; d1 < len; d1 += Vec::size()) {
    index0_vec.store(index_ptr + d1);
    out_vec.store(out_data + d1);
  }
  for (; d1 < size; d1++) {
    ind[d1] = ih0 * input_width + iw0;
    out_data[d1] = min_value;
  }
  // Pass II: compute local max
  for (int64_t id = id0; id < id1; id += dilationD) {
    for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
      for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
        const scalar_t* in = input_data + (n * input_depth * input_height * input_width +
            id * input_height * input_width + ih * input_width + iw) * channels;

        int64_t d2 = 0;
        for (; d2 < len; d2 += Vec::size()) {
          iVec index_vec = iVec(id * input_height * input_width + ih * input_width + iw);
          Vec val_vec = Vec::loadu(in + d2);
          iVec maxindex_vec = iVec::loadu(index_ptr + d2);
          Vec maxval_vec = Vec::loadu(out_data + d2);

          // true = all ones, false = all zeros
          Vec mask = (val_vec > maxval_vec) | is_nan_vec(val_vec);
          iVec imask = vec::cast<integer_t>(mask);
          Vec out_vec = Vec::blendv(maxval_vec, val_vec, mask);
          iVec ind_vec = iVec::blendv(maxindex_vec, index_vec, imask);

          out_vec.store(out_data + d2);
          ind_vec.store(index_ptr + d2);
        }
        for (; d2 < size; d2++) {
          int64_t index = id * input_height * input_width + ih * input_width + iw;
          scalar_t val = in[d2];
          int64_t maxindex = ind[d2];
          scalar_t maxval = out_data[d2];

          bool mask = (val > maxval) || is_nan(static_cast<double>(val));
          out_data[d2] = mask ? val : maxval;
          ind[d2] = mask ? index : maxindex;
        }
      }
    }
  }
}

// std::is_same<scalar_t, at::BFloat16> || std::is_same<scalar_t, at::Half>
template <typename scalar_t, typename opmath_t>
inline
std::enable_if_t<!std::is_same_v<scalar_t, opmath_t>, void>
compute_internal(
  const scalar_t* input_data,
  scalar_t* out_data,
  opmath_t* max_ptr,
  vec::int_same_size_t<opmath_t>* index_ptr,
  int64_t* ind,
  int64_t input_depth, int64_t input_height, int64_t input_width, int64_t channels,
  int64_t n,
  int64_t len,
  int64_t size,
  int64_t id0, int64_t id1,
  int64_t ih0, int64_t ih1,
  int64_t iw0, int64_t iw1,
  int64_t dilationD,
  int64_t dilationH,
  int64_t dilationW) {
  using Vec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<opmath_t>;
  using iVec = vec::Vectorized<int32_t>;
  // Pass I: init out lane
  iVec index0_vec = iVec(id0 * input_height * input_width + ih0 * input_width + iw0);
  fVec out_vec = fVec(-std::numeric_limits<opmath_t>::infinity());
  int64_t d1 = 0;
  for (; d1 < len; d1 += fVec::size()) {
    index0_vec.store(index_ptr + d1);
    out_vec.store(max_ptr + d1);
  }
  for (; d1 < size; d1++) {
    ind[d1] = ih0 * input_width + iw0;
    max_ptr[d1] = -std::numeric_limits<opmath_t>::infinity();
  }
  // Pass II: compute local max
  for (int64_t id = id0; id < id1; id += dilationD) {
    for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
      for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
        const scalar_t* in = input_data + (n * input_depth * input_height * input_width +
            id * input_height * input_width + ih * input_width + iw) * channels;

        int64_t d2 = 0;
        for (; d2 < len; d2 += Vec::size()) {
          iVec index_ivec = iVec(id * input_height * input_width + ih * input_width + iw);
          Vec val_bvec = Vec::loadu(in + d2);
          auto [val_fvec0, val_fvec1] = convert_to_float<scalar_t>(val_bvec);

          iVec maxindex_ivec0 = iVec::loadu(index_ptr + d2);
          iVec maxindex_ivec1 = iVec::loadu(index_ptr + d2 + iVec::size());
          fVec maxval_fvec0 = fVec::loadu(max_ptr + d2);
          fVec maxval_fvec1 = fVec::loadu(max_ptr + d2 + fVec::size());

          // true = all ones, false = all zeros
          fVec mask0 = (val_fvec0 > maxval_fvec0) | is_nan_vec(val_fvec0);
          fVec mask1 = (val_fvec1 > maxval_fvec1) | is_nan_vec(val_fvec1);
          iVec imask0 = vec::cast<int32_t>(mask0);
          iVec imask1 = vec::cast<int32_t>(mask1);

          fVec max_fvec0 = fVec::blendv(maxval_fvec0, val_fvec0, mask0);
          fVec max_fvec1 = fVec::blendv(maxval_fvec1, val_fvec1, mask1);
          iVec ind_vec0 = iVec::blendv(maxindex_ivec0, index_ivec, imask0);
          iVec ind_vec1 = iVec::blendv(maxindex_ivec1, index_ivec, imask1);

          max_fvec0.store(max_ptr + d2);
          max_fvec1.store(max_ptr + d2 + fVec::size());
          // out_vec.store(out + d2);
          ind_vec0.store(index_ptr + d2);
          ind_vec1.store(index_ptr + d2 + iVec::size());
        }
        for (; d2 < size; d2++) {
          int64_t index = id * input_height * input_width + ih * input_width + iw;
          opmath_t val = opmath_t(in[d2]);
          int64_t maxindex = ind[d2];
          opmath_t maxval = max_ptr[d2];

          bool mask = (val > maxval) || std::isnan(val);
          max_ptr[d2] = mask ? val : maxval;
          ind[d2] = mask ? index : maxindex;
        }
      }
    }
  }
  // Convert max values from float to bfloat16/half
  int64_t d3 = 0;
  for (; d3 < len; d3 += Vec::size()) {
    fVec max_fvec0 = fVec::loadu(max_ptr + d3);
    fVec max_fvec1 = fVec::loadu(max_ptr + d3 + fVec::size());
    Vec max_bvec = convert_from_float<scalar_t>(max_fvec0, max_fvec1);
    max_bvec.store(out_data + d3);
  }
  for (; d3 < size; d3++) {
    out_data[d3] = scalar_t(max_ptr[d3]);
  }
}

template <typename scalar_t, bool is_3d>
void cpu_max_pool(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    IntArrayRef kWHD,
    IntArrayRef dWHD,
    IntArrayRef padWHD,
    IntArrayRef dilWHD) {
  size_t dims =  is_3d ? 3 : 2;
  TORCH_CHECK(kWHD.size() == dims && dWHD.size() == dims && padWHD.size() == dims && dilWHD.size() == dims,
              "max pooling 2d/3d are not matched");
  int kW = kWHD[0];
  int kH = kWHD[1];
  int dW = dWHD[0];
  int dH = dWHD[1];
  int padW = padWHD[0];
  int padH = padWHD[1];
  int dilationW = dilWHD[0];
  int dilationH = dilWHD[1];

  int kD = is_3d ? kWHD[dims - 1] : 1;
  int dD = is_3d ? dWHD[dims - 1] : 1;
  int padD = is_3d ? padWHD[dims - 1] : 0;
  int dilationD = is_3d ? dilWHD[dims - 1] : 1;

  auto input = input_.contiguous();
  auto output = output_.contiguous();
  auto indices = indices_.contiguous();

  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

  int64_t ndim = input.ndimension();
  // treat batch size and channels as one dimension
  //
  // MaxPool2d:
  //   ndim == 3: CHW
  //   ndim == 4: NCHW
  //
  // MaxPool3d:
  //   ndim == 4: CDHW
  //   ndim == 5: NCDHW
  int64_t channels;
  if (is_3d) {
    channels = ndim == 4 ? input.size(0) : input.size(0) * input.size(1);
  } else {
    channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
  }
  int64_t input_depth = is_3d ? input.size(-3) : 1;
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_depth = is_3d ? output.size(-3) : 1;
  int64_t output_height = output.size(-2);
  int64_t output_width = output.size(-1);

  using opmath_t = at::opmath_type<scalar_t>;
  // parallel on dim N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      const scalar_t* input_ptr = input_data + c * input_depth * input_height * input_width;
      scalar_t* output_ptr = output_data + c * output_depth * output_height * output_width;
      int64_t* indices_ptr = indices_data + c * output_depth * output_height * output_width;

      for (int64_t od = 0; od < output_depth; od++) {
        int64_t id0 = od * dD - padD;
        int64_t id1 = std::min(id0 + (kD - 1) * dilationD + 1, input_depth);
        while(id0 < 0) { id0 += dilationD; }

        for (int64_t oh = 0; oh < output_height; oh++) {
          int64_t ih0 = oh * dH - padH;
          int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
          while(ih0 < 0) { ih0 += dilationH; }

          for (int64_t ow = 0; ow < output_width; ow++) {
            int64_t iw0 = ow * dW - padW;
            int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
            while(iw0 < 0) { iw0 += dilationW; }

            // compute local max
            int64_t maxindex = id0 * input_height * input_width + ih0 * input_width + iw0;
            opmath_t maxval;
            if (std::numeric_limits<opmath_t>::has_infinity) {
              maxval = -std::numeric_limits<opmath_t>::infinity();
            } else {
              maxval = std::numeric_limits<opmath_t>::min();
            }

            for (int64_t id = id0; id < id1; id += dilationD) {
              for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
                for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
                  int64_t index = id * input_height * input_width + ih * input_width + iw;
                  opmath_t val = input_ptr[index];
                  if ((val > maxval) || is_nan(static_cast<double>(val))) {
                    maxval = val;
                    maxindex = index;
                  }
                }
              }
            }

            // set output to local max and store location of max
            int64_t i = od * output_height * output_width + oh * output_width + ow;
            output_ptr[i] = scalar_t(maxval);
            indices_ptr[i] = maxindex;
          }
        }
      }
    }
  });

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
  if (!indices_.is_contiguous()) {
    indices_.copy_(indices);
  }
}

template <typename scalar_t, bool is_3d>
void cpu_max_pool_channels_last(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    IntArrayRef kWHD,
    IntArrayRef dWHD,
    IntArrayRef padWHD,
    IntArrayRef dilWHD) {
  size_t dims =  is_3d ? 3 : 2;
  TORCH_CHECK(kWHD.size() == dims && dWHD.size() == dims && padWHD.size() == dims && dilWHD.size() == dims,
              "max pooling 2d/3d are not matched");
  int64_t ndim = input_.ndimension();
  // MaxPool2d: NHWC
  // MaxPool3d: NDHWC
  if (is_3d) {
    TORCH_CHECK(ndim == 5, "max pooling 3d with channels last format supports tensors with 5 dims");
  } else {
    TORCH_CHECK(ndim == 4, "max pooling 2d with channels last format supports tensors with 4 dims");
  }

  int kW = kWHD[0];
  int kH = kWHD[1];
  int dW = dWHD[0];
  int dH = dWHD[1];
  int padW = padWHD[0];
  int padH = padWHD[1];
  int dilationW = dilWHD[0];
  int dilationH = dilWHD[1];

  int kD = is_3d ? kWHD[dims - 1] : 1;
  int dD = is_3d ? dWHD[dims - 1] : 1;
  int padD = is_3d ? padWHD[dims - 1] : 0;
  int dilationD = is_3d ? dilWHD[dims - 1] : 1;

  auto memory_format = is_3d ? at::MemoryFormat::ChannelsLast3d : at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = is_3d ? input.size(-3) : 1;
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_depth = is_3d ? output.size(-3) : 1;
  int64_t output_height = output.size(-2);
  int64_t output_width = output.size(-1);

  using opmath_t = at::opmath_type<scalar_t>;
  using Vec = vec::Vectorized<scalar_t>;
  using integer_t = vec::int_same_size_t<opmath_t>;
  // for the convenience of vectorization, use integer of the same size of scalar_t,
  //   e.g. int32_t for float, int64_t for double
  // need to make sure doesn't overflow
  TORCH_CHECK(input_depth * input_height * input_width <= std::numeric_limits<integer_t>::max());

  // parallel on dim N, {D}, H, W
  at::parallel_for(0, nbatch * output_depth * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, od, output_depth, oh, output_height, ow, output_width);

    int64_t size = channels;
    int64_t len = size - (size % Vec::size());
    // temp buffer holding index with integer_t
    auto index_buffer = std::make_unique<integer_t []>(len);
    integer_t * index_ptr = index_buffer.get();
    // temp buffer holding max value with opmath_t
    std::unique_ptr<opmath_t []> max_arr;
    opmath_t* max_ptr = nullptr;
    if (!std::is_same_v<scalar_t, opmath_t>) {
      max_arr = std::make_unique<opmath_t[]>(size);
      max_ptr = max_arr.get();
    }

    for (int64_t i = begin; i < end; i++) {
      int64_t id0 = od * dD - padD;
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      int64_t id1 = std::min(id0 + (kD - 1) * dilationD + 1, input_depth);
      int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
      int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
      while(id0 < 0) { id0 += dilationD; }
      while(ih0 < 0) { ih0 += dilationH; }
      while(iw0 < 0) { iw0 += dilationW; }

      scalar_t* out = output_data + i * channels;
      int64_t* ind = indices_data + i * channels;

      compute_internal(input_data, out, max_ptr, index_ptr, ind, input_depth, input_height, input_width, channels,
                        n, len, size, id0, id1, ih0, ih1, iw0, iw1,
                        dilationD, dilationH, dilationW);

      // convert indice data type
      vec::convert<integer_t, int64_t>(index_buffer.get(), ind, len);

      // move on to next output index
      data_index_step(n, nbatch, od, output_depth, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
  if (!indices_.is_contiguous(memory_format)) {
    indices_.copy_(indices);
  }
}


template <typename scalar_t, bool is_3d>
void cpu_max_pool_backward(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const Tensor& indices_) {
  auto grad_output = grad_output_.contiguous();
  auto indices = indices_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto indices_data = indices.const_data_ptr<int64_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  // treat batch size and channels as one dimension
  //
  // MaxPool2d:
  //   ndim == 3: CHW
  //   ndim == 4: NCHW
  //
  // MaxPool3d:
  //   ndim == 4: CDHW
  //   ndim == 5: NCDHW
  int64_t ndim = grad_output.ndimension();
  int64_t channels;
  if (is_3d) {
    channels = ndim == 4 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  } else {
    channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  }
  int64_t input_depth = is_3d ? grad_input.size(-3) : 1;

  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_depth = is_3d ? grad_output.size(-3) : 1;
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // parallel on dim of N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      scalar_t* grad_input_ptr = grad_input_data + c * input_depth * input_height * input_width;
      const scalar_t* grad_output_ptr = grad_output_data + c * output_depth * output_height * output_width;
      const int64_t * indices_ptr = indices_data + c * output_depth * output_height * output_width;

      for (int64_t od = 0; od < output_depth; od++) {
        for (int64_t oh = 0; oh < output_height; oh++) {
          for (int64_t ow = 0; ow < output_width; ow++) {
            // retrieve position of max
            int64_t index = od * output_height * output_width + oh * output_width + ow;
            int64_t maxindex = indices_ptr[index];
            if (maxindex != -1) {
              // update gradient
              grad_input_ptr[maxindex] += grad_output_ptr[index];
            }
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t, bool is_3d>
void cpu_max_pool_backward_channels_last(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const Tensor& indices_) {
  int64_t ndim = grad_output_.ndimension();
  if (is_3d) {
    TORCH_CHECK(ndim == 5, "MaxPool3d backward with channels last format supports tensors with 5 dims.");
  } else {
    TORCH_CHECK(ndim == 4, "MaxPool2d backward with channels last format supports tensors with 4 dims.");
  }
  auto memory_format = is_3d ? at::MemoryFormat::ChannelsLast3d
                             : at::MemoryFormat::ChannelsLast;
  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto indices_data = indices.const_data_ptr<int64_t>();

  // MaxPool2d: NHWC
  // MaxPool3d: NDHWC
  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_depth = is_3d ? grad_input.size(2) : 1;
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_depth = is_3d ? grad_output.size(2) : 1;
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // parallel on dim N
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {
      scalar_t* grad_input_ptr = grad_input_data + n * input_depth * input_height * input_width * channels;
      const scalar_t* grad_output_ptr = grad_output_data + n * output_depth * output_height * output_width * channels;
      const int64_t* indices_ptr = indices_data + n * output_depth * output_height * output_width * channels;

      for (int64_t od = 0; od < output_depth; od++) {
        for (int64_t oh = 0; oh < output_height; oh++) {
          for (int64_t ow = 0; ow < output_width; ow++) {
            const scalar_t* gout = grad_output_ptr + (od * output_height * output_width + oh * output_width + ow) * channels;
            const int64_t* ind = indices_ptr + (od * output_height * output_width + oh * output_width + ow) * channels;
            // TODO: gcc vectorization
            for (int64_t c = 0; c < channels; c++) {
              int64_t maxindex = ind[c];
              if (maxindex != -1) {
                grad_input_ptr[maxindex * channels + c] += gout[c];
              }
            }
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous(memory_format)) {
    grad_input_.copy_(grad_input);
  }
}

void max_pool2d_kernel_impl(
    const Tensor& output,
    const Tensor& indices,
    const Tensor& input,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "max_pool2d", [&] {
        cpu_max_pool<scalar_t, /*is 3d*/false>(output, indices, input, {kW, kH}, {dW, dH}, {padW, padH}, {dilationW, dilationH});
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "max_pool2d_channels_last", [&] {
        cpu_max_pool_channels_last<scalar_t, false>(output, indices, input, {kW, kH}, {dW, dH}, {padW, padH}, {dilationW, dilationH});
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void max_pool3d_kernel_impl(
    Tensor& output,
    Tensor& indices,
    const Tensor& input,
    int kW, int kH, int kD,
    int dW, int dH, int dD,
    int padW, int padH, int padD,
    int dilationW, int dilationH, int dilationD) {
  if (input.ndimension() == 4) {
    Tensor input_cl_check = input.unsqueeze(0);
    // align with cuda:
    // work around buggy behavior of suggest_memory_format here where
    // suggested format of unsqueezed tensor is contiguous while it is
    // really only contiguous in ChannelsLast3d
    if ((!input_cl_check.is_contiguous()) &&
                     input_cl_check.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
      TORCH_CHECK(output.ndimension() == 4 && indices.ndimension() == 4);
      DimVector out_sizes(output.sizes().begin(), output.sizes().end());
      out_sizes.insert(out_sizes.begin(), 1);
      output.resize_(out_sizes, at::MemoryFormat::ChannelsLast3d);
      DimVector indices_sizes(indices.sizes().begin(), indices.sizes().end());
      indices_sizes.insert(indices_sizes.begin(), 1);
      indices.resize_(indices_sizes, at::MemoryFormat::ChannelsLast3d);
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "max_pool3d_channels_last", [&] {
        cpu_max_pool_channels_last<scalar_t, /*is 3d*/true>(output, indices, input_cl_check,
          {kW, kH, kD}, {dW, dH, dD}, {padW, padH, padD}, {dilationW, dilationH, dilationD});
      });
      output.squeeze_(0);
      indices.squeeze_(0);
      return;
    }
  }
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "max_pool3d", [&] {
        cpu_max_pool<scalar_t, /*is 3d*/true>(output, indices, input,
            {kW, kH, kD}, {dW, dH, dD}, {padW, padH, padD}, {dilationW, dilationH, dilationD});
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "max_pool3d_channels_last", [&] {
        cpu_max_pool_channels_last<scalar_t, true>(output, indices, input,
          {kW, kH, kD}, {dW, dH, dD}, {padW, padH, padD}, {dilationW, dilationH, dilationD});
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

void max_pool2d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& indices) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "max_pool2d_backward", [&] {
        cpu_max_pool_backward<scalar_t, /*is 3d*/ false>(grad_input, grad_output, indices);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "max_pool2d_backward_channels_last", [&] {
        cpu_max_pool_backward_channels_last<scalar_t, /*is 3d*/ false>(grad_input, grad_output, indices);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void max_pool3d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& indices) {
  if (grad_output.ndimension() == 4) {
    Tensor grad_output_cl_check = grad_output.unsqueeze(0);
    // align with cuda:
    // work around buggy behavior of suggest_memory_format here where
    // suggested format of unsqueezed tensor is contiguous while it is
    // really only contiguous in ChannelsLast3d
    if ((!grad_output_cl_check.is_contiguous()) &&
                     grad_output_cl_check.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
      TORCH_CHECK(grad_input.ndimension() == 4 && indices.ndimension() == 4);
      DimVector sizes(grad_input.sizes().begin(), grad_input.sizes().end());
      sizes.insert(sizes.begin(), 1);
      grad_input.resize_(sizes, at::MemoryFormat::ChannelsLast3d);
      auto _indices = indices.unsqueeze(0).contiguous(at::MemoryFormat::ChannelsLast3d);
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "max_pool3d_backward_channels_last", [&] {
        cpu_max_pool_backward_channels_last<scalar_t, /*is_3d*/ true>(grad_input, grad_output_cl_check, _indices);
      });
      grad_input.squeeze_(0);
      return;
    }
  }
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "max_pool3d_backward", [&] {
        cpu_max_pool_backward<scalar_t, /*is_3d*/ true>(grad_input, grad_output, indices);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "max_pool3d_backward_channels_last", [&] {
        cpu_max_pool_backward_channels_last<scalar_t, /*is_3d*/ true>(grad_input, grad_output, indices);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

} // anonymous namespace

REGISTER_DISPATCH(max_pool2d_kernel, &max_pool2d_kernel_impl)
REGISTER_DISPATCH(max_pool2d_backward_kernel, &max_pool2d_backward_kernel_impl)
REGISTER_DISPATCH(max_pool3d_kernel, &max_pool3d_kernel_impl)
REGISTER_DISPATCH(max_pool3d_backward_kernel, &max_pool3d_backward_kernel_impl)
} // at::native
