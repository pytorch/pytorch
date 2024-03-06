#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/MatrixRef.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/RNN.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cudnn_init_dropout_state.h>
#include <ATen/ops/_cudnn_init_dropout_state_native.h>
#include <ATen/ops/_cudnn_rnn.h>
#include <ATen/ops/_cudnn_rnn_backward_native.h>
#include <ATen/ops/_cudnn_rnn_flatten_weight_native.h>
#include <ATen/ops/_cudnn_rnn_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#if !AT_CUDNN_ENABLED()

namespace at {
namespace native {

// See Note [ATen preprocessor philosophy]

Tensor _cudnn_rnn_flatten_weight(
    TensorList weight_arr,
    int64_t weight_stride0,
    int64_t input_size,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    bool fn_bidirectional) {
  AT_ERROR("_cudnn_rnn_flatten_weight: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _cudnn_rnn(
    const Tensor& input_r,
    TensorList weight,
    int64_t weight_stride0,
    const c10::optional<Tensor>& weight_buf_r_opt,
    const Tensor& hx,
    const c10::optional<Tensor>& cx_opt,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    double fn_dropout,
    bool fn_train,
    bool fn_bidirectional,
    IntArrayRef fn_batch_sizes,
    const c10::optional<Tensor>& fn_dropout_state_opt) {
  AT_ERROR("_cudnn_rnn: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _cudnn_rnn_backward(
    const Tensor& input,
    TensorList weight,
    int64_t weight_stride0,
    const Tensor& weight_buf,
    const Tensor& hx,
    const c10::optional<Tensor>& cx_opt,
    const Tensor& output,
    const c10::optional<Tensor>& grad_output_r_opt,
    const c10::optional<Tensor>& grad_hy_r_opt,
    const c10::optional<Tensor>& grad_cy_r_opt,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    double dropout,
    bool train,
    bool bidirectional,
    IntArrayRef batch_sizes,
    const c10::optional<Tensor>& dropout_state_opt,
    const Tensor& reserve,
    std::array<bool, 4> output_mask) {
  AT_ERROR("_cudnn_rnn_backward: ATen not compiled with cuDNN support");
}

Tensor _cudnn_init_dropout_state(
    double dropout,
    bool train,
    int64_t dropout_seed,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
      pin_memory);

  AT_ERROR("_cudnn_init_dropout_state: ATen not compiled with cuDNN support");
}

} // namespace native
} // namespace at

#else // AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/RNNUtils.h>

namespace at {
namespace native {

namespace {
// DropoutDescriptor

struct DropoutDescriptorParams {
  bool train;
  double dropout;
  Tensor dropout_state;
  DropoutDescriptorParams() = default;
  void set(bool train_, double dropout_, Tensor dropout_state_) {
    train = train_;
    dropout = dropout_;
    dropout_state = dropout_state_;
  }
  DropoutDescriptor descriptor(cudnnHandle_t handle) const {
    auto dropout_p = train ? dropout : 0;
    DropoutDescriptor dropout_desc;
    if (dropout_p == 0) {
      dropout_desc.set_no_dropout(handle);
    } else {
      dropout_desc.set(handle, dropout_p, dropout_state);
    }
    return dropout_desc;
  }
};

// RNNDescriptor

struct RNNDescriptorParams {
#ifdef USE_CUDNN_RNN_V8_API
  int64_t input_size;
  bool packed;
#endif
  int64_t hidden_size;
  int64_t proj_size;
  int64_t num_layers;
  cudnnDirectionMode_t bidirectional;
  cudnnRNNMode_t mode;
  cudnnDataType_t datatype;
  cudnnDataType_t input_datatype;
  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;

  int64_t num_directions() const {
    return bidirectional ? 2 : 1;
  }

  void set_mode(int64_t fn_mode) {
    switch (fn_mode) {
      case CUDNN_RNN_RELU:
        mode = CUDNN_RNN_RELU;
        break;
      case CUDNN_RNN_TANH:
        mode = CUDNN_RNN_TANH;
        break;
      case CUDNN_LSTM:
        mode = CUDNN_LSTM;
        break;
      case CUDNN_GRU:
        mode = CUDNN_GRU;
        break;
      default: {
        std::ostringstream oss;
        oss << "unrecognized cuDNN RNN mode " << fn_mode;
        AT_ERROR(oss.str());
      }
    }
  }

  void set_bidirectional(bool fn_bidirectional) {
    bidirectional =
        fn_bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
  }

  void set_algo(cudnnRNNAlgo_t algo) {
    this->algo = algo;
  }

#ifndef USE_CUDNN_RNN_V8_API
  void set(
      int64_t mode,
      int64_t hidden_size,
      int64_t proj_size,
      int64_t num_layers,
      bool bidirectional,
      cudnnDataType_t datatype,
      cudnnDataType_t input_datatype){
#else
  void set(
      int64_t mode,
      int64_t input_size,
      bool packed,
      int64_t hidden_size,
      int64_t proj_size,
      int64_t num_layers,
      bool bidirectional,
      cudnnDataType_t datatype,
      cudnnDataType_t input_datatype) {
#endif
      this->set_mode(mode);
#ifdef USE_CUDNN_RNN_V8_API
  this->input_size = input_size;
  this->packed = packed;
#endif
  this->hidden_size = hidden_size;
  this->proj_size = proj_size;
  this->num_layers = num_layers;
  this->set_bidirectional(bidirectional);
  this->datatype = datatype;
  this->input_datatype = input_datatype;
}

RNNDescriptor
descriptor(cudnnHandle_t handle, DropoutDescriptor&& dropout_desc) const {
  RNNDescriptor rnn_desc;
#ifndef USE_CUDNN_RNN_V8_API
  rnn_desc.set(
      handle,
      hidden_size,
      proj_size,
      num_layers,
      std::move(dropout_desc),
      input_mode,
      bidirectional,
      mode,
      datatype,
      input_datatype,
      algo,
      at::globalContext().allowTF32CuDNN());
#else
    rnn_desc.set(
        handle,
        input_size,
        packed,
        hidden_size,
        proj_size,
        num_layers,
        std::move(dropout_desc),
        input_mode,
        bidirectional,
        mode,
        datatype,
        input_datatype,
        algo,
        at::globalContext().allowTF32CuDNN());
#endif
  return rnn_desc;
}

// In some cases, a use of RNNDescriptor does not rely on the
// DropoutDescriptor.  In this case, we fake up a no-dropout
// descriptor to make the RNN descriptor initialization go through.
// This is used by _cudnn_rnn_flatten_weight, which needs an
// RNNDescriptor for get_parameters(), but does not actually need
// a fully initialized dropout descriptor.  This lets us avoid
// having to pass the dropout state to flatten, which has no business
// knowing what the dropout state is.
RNNDescriptor descriptor(cudnnHandle_t handle) const {
  DropoutDescriptor dropout_desc;
  dropout_desc.set_no_dropout(handle);
  return descriptor(handle, std::move(dropout_desc));
}
}; // namespace

// TensorDescriptor list
#ifndef USE_CUDNN_RNN_V8_API
std::vector<TensorDescriptor> rnn_descriptor_sequence(
    const Tensor& tensor,
    IntArrayRef batch_sizes) {
  std::vector<TensorDescriptor> descriptors(batch_sizes.size());
  size_t i = 0;
  // To be mutated in the loop
  auto batch_tensor_size = tensor.sizes().vec();
  for (auto batch_size : batch_sizes) {
    batch_tensor_size[0] = batch_size;
    // NB: cuDNN RNN API does not support 2d descriptors, so we
    // must pad it out to 3d.
    descriptors[i].set(
        getCudnnDataType(tensor), batch_tensor_size, tensor.strides(), 3);
    i++;
  }
  return descriptors;
}

std::vector<TensorDescriptor> rnn_descriptor(const Tensor& tensor, int64_t N) {
  std::vector<TensorDescriptor> descriptors(N);
  for (const auto i : c10::irange(N)) {
    descriptors[i].set(tensor, 5);
  }
  return descriptors;
}
#else
auto rnn_descriptor_sequence(
    const Tensor& tensor,
    uint32_t batch_size,
    const IntArrayRef batch_sizes,
    uint32_t seq_len,
    uint32_t vector_size) { // packed case
  RNNDataDescriptor r;
  std::vector<int> seqLengthArray(batch_size, 1);
  // cuDNN wants the sequence lengths for a packed batch as if they
  // were unpacked, e.g., for the
  // Sequence 1: ABCD
  // Sequence 2: EF
  // Sequence 3: G
  // case below, this would be [4, 2, 1] (has length == mini_batch)
  // TODO(eqy): There's probably a smarter way to do this than O(SN)
  for (auto it = batch_sizes.begin(); it != batch_sizes.end(); it++) {
    // everyone starts at sequence length 1 so we skip an iteration
    if (it == batch_sizes.begin()) {
      continue;
    }
    for (const auto idx : c10::irange(*it)) {
      seqLengthArray[idx]++;
    }
  }
  r.set(
      tensor,
      CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
      seq_len,
      batch_size,
      vector_size,
      seqLengthArray.data());
  return r;
}

auto rnn_descriptor(
    const Tensor& tensor,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t vector_size) {
  RNNDataDescriptor r;
  // NB: Looks like even if batch_first is true here we always want
  // SEQ_MAJOR_UNPACKED, because the input appears to be transposed if it is
  // barch-major
  const auto layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
  std::vector<int32_t> seqLengthArray(batch_size, seq_len);
  r.set(
      tensor, layout, seq_len, batch_size, vector_size, seqLengthArray.data());
  return r;
}
#endif

// The best way to understand the meaning of the values stored in
// this struct is to consider each of the possible ways our
// input can be structured.
//
// Suppose you want to run RNN on the following variable
// length inputs:
//
//    Sequence 1: ABCD
//    Sequence 2: EF
//    Sequence 3: G
//
// (Let _ be padding when we have non-packed representations.)
//
// # Packed input (batch_sizes is non-empty)
//
//  input_size
// +------+                    +
// | A    |                    |
// | E    | mini_batch =       |
// | G    | batch_sizes[0] = 3 |
// +------+                    |
// | B    |                    | batch_sizes_sum = 7
// | F    | batch_sizes[1] = 2 |
// +------+                    |
// | C    | batch_sizes[2] = 1 |
// +------+                    |
// | D    | batch_sizes[3] = 1 |
// +------+                    +
//
//              (seq_length = 4)
//
//    input.size() = batch_sizes_sum x input_size
//
// # Unpacked input (batch_first = false)
//
//  mini_batch = 3
// +-------+
// | A E G |
// | B F _ | seq_length = 4
// | C _ _ |
// | D _ _ |
// +-------+
//    ...    input_size
// +-------+
//
//    input.size() = seq_length x mini_batch x input_size
//
// # Unpacked input (batch_first = true)
//
//  seq_length = 4
// +---------+
// | A B C D |
// | E F _ _ | mini_batch = 3
// | G _ _ _ |
// +---------+
//     ...     input_size
// +---------+
//
//    input.size() = mini_batch x seq_length x input_size
//
struct TensorDescriptorListParams {
  IntArrayRef batch_sizes;
  int64_t seq_length;
  int64_t mini_batch;
  // NB: this is not input.size(), which is an IntArrayRef; instead, this
  // size of the inner-most dimension.  In NL applications, this is usually
  // the size of the embedding.  You can also think of this as the size
  // of the "channel" dimension (at risk of confusing vision researchers :)
  int64_t input_size;
  // Only valid when !is_input_packed
  int64_t batch_sizes_sum; // == sum(batch_sizes)

  bool is_input_packed() const {
    return batch_sizes.size() != 0;
  }

  void set(
      IntArrayRef input_sizes,
      IntArrayRef batch_sizes_,
      bool batch_first) {
    batch_sizes = batch_sizes_;
    if (is_input_packed()) {
      seq_length = batch_sizes.size();
      mini_batch = batch_sizes[0];
      // NB: When input is packed, the mini_batch size is NOT the size
      // of the outer dimension
      batch_sizes_sum = input_sizes[0];
      input_size = input_sizes[1];
    } else {
      if (batch_first) {
        seq_length = input_sizes[1];
        mini_batch = input_sizes[0];
      } else {
        seq_length = input_sizes[0];
        mini_batch = input_sizes[1];
      }
      input_size = input_sizes[2];
      // TODO: Actually, would this make ASAN's job harder catching
      // an uninitialized access?
      batch_sizes_sum = -1; // something bogus in case we access it
    }
  }
#ifndef USE_CUDNN_RNN_V8_API
  // TODO: check x for consistency with input_size?
  std::vector<TensorDescriptor> descriptors(Tensor x) const {
    auto is_input_packed = batch_sizes.size() != 0;
    if (is_input_packed) {
      return rnn_descriptor_sequence(x, batch_sizes);
    } else {
      return rnn_descriptor(x[0], seq_length);
    }
  }
#else
  auto descriptors(Tensor x) const {
    auto is_input_packed = batch_sizes.size() != 0;
    if (is_input_packed) {
      return rnn_descriptor_sequence(
          x, mini_batch, batch_sizes, seq_length, x.size(-1));
    } else {
      return rnn_descriptor(x, mini_batch, seq_length, x.size(-1));
    }
  }
#endif
};

// Everything together

struct RNNParams {
  DropoutDescriptorParams dropout;
  RNNDescriptorParams rnn;
  TensorDescriptorListParams tensors;
};

// NB: Doesn't include the weight descriptor
struct RNNDescriptors {
  RNNDescriptor rnn_desc;
  // NB: this won't actually lay out the tensor descriptor pointers
  // in the right way, so you'll have to preprocess them
#ifndef USE_CUDNN_RNN_V8_API
  std::vector<TensorDescriptor> x_descs;
  std::vector<TensorDescriptor> y_descs;
#else
  RNNDataDescriptor x_descs;
  RNNDataDescriptor y_descs;
#endif
  TensorDescriptor hx_desc;
  TensorDescriptor hy_desc;
  TensorDescriptor cx_desc;
  TensorDescriptor cy_desc;

  RNNDescriptors(
      const RNNParams& fn,
      cudnnHandle_t handle,
      Tensor x,
      Tensor y,
      Tensor hx,
      Tensor cx) {
    rnn_desc = fn.rnn.descriptor(handle, fn.dropout.descriptor(handle));
    x_descs = fn.tensors.descriptors(x);
    y_descs = fn.tensors.descriptors(y);
    hx_desc.set(hx, 5);
    hy_desc.set(hx, 5);
    if (cx.defined()) {
      cx_desc.set(cx, 5);
      cy_desc.set(cx, 5);
    }
  }

  // TODO: This is annoying, having to put the cudnnTensorDescriptor_t
  // in a contiguous array...
  std::vector<cudnnTensorDescriptor_t> get_descs(
      const std::vector<TensorDescriptor>& descs) {
    std::vector<cudnnTensorDescriptor_t> r;
    r.reserve(descs.size());
    for (auto& desc : descs) {
      r.emplace_back(desc.desc());
    }
    return r;
  }
#ifndef USE_CUDNN_RNN_V8_API
  std::vector<cudnnTensorDescriptor_t> get_x_descs() {
    return get_descs(x_descs);
  }

  std::vector<cudnnTensorDescriptor_t> get_y_descs() {
    return get_descs(y_descs);
  }
#endif
};

int64_t get_num_weights(
    cudnnHandle_t handle,
    const RNNDescriptor& rnn_desc,
#ifndef USE_CUDNN_RNN_V8_API
    const TensorDescriptor& x_desc,
#endif
    cudnnDataType_t datatype) {
  size_t weight_size;
#ifndef USE_CUDNN_RNN_V8_API
  AT_CUDNN_CHECK(cudnnGetRNNParamsSize(
      handle, rnn_desc.desc(), x_desc.desc(), &weight_size, datatype));
#else
  AT_CUDNN_CHECK(
      cudnnGetRNNWeightSpaceSize(handle, rnn_desc.desc(), &weight_size));
#endif
  auto elem_size = dataSize(datatype);
  TORCH_INTERNAL_ASSERT(
      weight_size % elem_size == 0,
      "cudnnGetRNNParamsSize returned nonsensical weight_size");
  return weight_size / elem_size;
}

int64_t _num_linear_layers(cudnnRNNMode_t mode) {
  switch (mode) {
    case CUDNN_LSTM:
      return 8;
    case CUDNN_GRU:
      return 6;
    case CUDNN_RNN_RELU:
      return 2;
    case CUDNN_RNN_TANH:
      return 2;
    default:
      AT_ERROR("unknown cuDNN RNN mode ", mode);
  }
}

void add_projection_weights(
    cudnnHandle_t handle,
    const RNNDescriptor& rnn_desc,
#ifndef USE_CUDNN_RNN_V8_API
    const TensorDescriptor& x_desc,
    const FilterDescriptor& w_desc,
#endif
    const Tensor& weight_buf,
    int64_t layer,
    std::vector<Tensor>& params) {
  void* matrix_pointer = nullptr;
  // assuming it's LSTM which has 8 "linear layers" (i.e. 4 weights and 4
  // biases)
  int64_t linear_id = 8;
#ifndef USE_CUDNN_RNN_V8_API
  FilterDescriptor lin_layer_mat_desc;
  AT_CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(
      /*handle=*/handle,
      /*rnnDesc=*/rnn_desc.desc(),
      /*layer=*/layer,
      /*xDesc=*/x_desc.desc(),
      /*wDesc=*/w_desc.desc(),
      /*w=*/weight_buf.data_ptr(),
      /*linLayerID=*/linear_id,
      /*linLayerMatDesc=*/lin_layer_mat_desc.mut_desc(),
      /*linLayerMat=*/&matrix_pointer));
#else
  void* unused_pointer;
  TensorDescriptor unused_desc;
  TensorDescriptor lin_layer_mat_desc;
  AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
      /*handle=*/handle,
      /*rnnDesc=*/rnn_desc.desc(),
      /*layer=*/layer,
      /*wDesc=*/weight_buf.numel() * weight_buf.element_size(),
      /*w=*/weight_buf.data_ptr(),
      /*linLayerID=*/linear_id,
      /*linLayerMatDesc=*/lin_layer_mat_desc.mut_desc(),
      /*linLayerMat=*/&matrix_pointer,
      unused_desc.mut_desc(),
      &unused_pointer));
#endif

  cudnnDataType_t data_type;
#ifndef USE_CUDNN_RNN_V8_API
  cudnnTensorFormat_t format;
#else
  int stride_dim_a[5];
#endif
  int nb_dims;
  constexpr int min_dim = 3;
  int filter_dim_a[min_dim];
#ifndef USE_CUDNN_RNN_V8_API
  AT_CUDNN_CHECK(cudnnGetFilterNdDescriptor(
      lin_layer_mat_desc.desc(),
      min_dim,
      &data_type,
      &format,
      &nb_dims,
      filter_dim_a));
#else
  AT_CUDNN_CHECK(cudnnGetTensorNdDescriptor(
      lin_layer_mat_desc.desc(),
      min_dim,
      &data_type,
      &nb_dims,
      filter_dim_a,
      stride_dim_a));
#endif

  TORCH_INTERNAL_ASSERT(
      nb_dims <= min_dim, "nb_dims = ", nb_dims, "; min_dim  = ", min_dim);
  auto elem_size = dataSize(getCudnnDataType(weight_buf));
  auto offset_bytes = (char*)matrix_pointer - (char*)weight_buf.data_ptr();
  TORCH_INTERNAL_ASSERT(
      offset_bytes % elem_size == 0,
      "offset_bytes = ",
      offset_bytes,
      "; elem_size = ",
      elem_size);
  size_t offset = offset_bytes / elem_size;

  int mat_numel = c10::multiply_integers(filter_dim_a, filter_dim_a + nb_dims);
  // Generate a new parameter tensor which is a view into the weight_buf.
  std::initializer_list<int64_t> size = {mat_numel, 1};
  Tensor param = at::empty({0}, weight_buf.options())
                     .set_(weight_buf.storage(), offset, size);
  params.emplace_back(std::move(param));
}

/*
  Returns weight and bias tensors for each layer of the RNN. These tensors
  are views on the underlying weight buffer allocated by CuDNN.

  Note: for LSTM and GRU, which have multiple parameters of each type (4 and 3,
  respectively), these parameters are concatenated along the first dimension.
        These parameters are returned in a consistent order by CuDNN:
            (reset, forget, cell, output) for LSTM
            (reset, input, new) for GRU
  Args:
      fn: The RNN function object holding the RNN state
      handle: a CuDNN handle
      weight_buf: a 1D tensor containing the CuDNN-allocated weight (or
  grad_weight) buffer Returns: parameters: [(weight_ih, weight_hh, bias_ih,
  bias_hh)*], with length equal to the num_layers. This is represented as a pair
  of vector, and outer-dimension stride (NB: Can't return MatrixRef because we
  need to allocate the underlying tensor)
*/
std::pair<std::vector<Tensor>, size_t> // stride0
get_parameters(
    cudnnHandle_t handle,
    const RNNDescriptorParams& rnn,
    const RNNDescriptor& rnn_desc,
#ifndef USE_CUDNN_RNN_V8_API
    const TensorDescriptor& x_desc,
    const FilterDescriptor& w_desc,
#endif
    const Tensor& weight_buf,
    bool include_bias = true) {
#ifndef USE_CUDNN_RNN_V8_API
  auto cudnn_methods = {
      cudnnGetRNNLinLayerMatrixParams, cudnnGetRNNLinLayerBiasParams};
#else
  auto cudnn_methods = {true, false};
#endif
  std::vector<Tensor> params;
  int64_t num_linear_layers = _num_linear_layers(rnn.mode);
  int64_t num_layers = rnn.num_directions() * rnn.num_layers;
  size_t cur_offset = 0;
  size_t global_layer_params_count = 0;
  for (const auto layer : c10::irange(num_layers)) {
    size_t layer_params_count = 0;
    for (auto cudnn_method : cudnn_methods) {
      for (const auto linear_id : c10::irange(num_linear_layers)) {
        void* matrix_pointer;
#ifndef USE_CUDNN_RNN_V8_API
        FilterDescriptor lin_layer_mat_desc;
        AT_CUDNN_CHECK(cudnn_method(
            handle,
            rnn_desc.desc(),
            layer,
            x_desc.desc(),
            w_desc.desc(),
            weight_buf.data_ptr(),
            linear_id,
            lin_layer_mat_desc.mut_desc(),
            &matrix_pointer));
#else
        void* unused_pointer = nullptr;
        TensorDescriptor unused_desc;
        TensorDescriptor lin_layer_mat_desc;
        for (int stateless = 0; stateless < 100; stateless++) {
          if (cudnn_method) { // matrix
            AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
                handle,
                rnn_desc.desc(),
                layer,
                weight_buf.numel() * weight_buf.element_size(),
                weight_buf.data_ptr(),
                linear_id,
                lin_layer_mat_desc.mut_desc(),
                &matrix_pointer,
                unused_desc.mut_desc(),
                &unused_pointer));
          } else { // bias
            AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
                handle,
                rnn_desc.desc(),
                layer,
                weight_buf.numel() * weight_buf.element_size(),
                weight_buf.data_ptr(),
                linear_id,
                unused_desc.mut_desc(),
                &unused_pointer,
                lin_layer_mat_desc.mut_desc(),
                &matrix_pointer));
          }
        }
#endif
        cudnnDataType_t data_type;
#ifndef USE_CUDNN_RNN_V8_API
        cudnnTensorFormat_t format;
#else
        int stride_dim_a[5];
#endif
        int nb_dims;
        constexpr int min_dim = 3;
        int filter_dim_a[min_dim];
#ifndef USE_CUDNN_RNN_V8_API
        AT_CUDNN_CHECK(cudnnGetFilterNdDescriptor(
            lin_layer_mat_desc.desc(),
            min_dim,
            &data_type,
            &format,
            &nb_dims,
            filter_dim_a));
#else
        AT_CUDNN_CHECK(cudnnGetTensorNdDescriptor(
            lin_layer_mat_desc.desc(),
            min_dim,
            &data_type,
            &nb_dims,
            filter_dim_a,
            stride_dim_a));
#endif

        TORCH_INTERNAL_ASSERT(
            nb_dims <= min_dim,
            "nb_dims = ",
            nb_dims,
            "; min_dim  = ",
            min_dim);
        auto elem_size = dataSize(getCudnnDataType(weight_buf));
        auto offset_bytes =
            (char*)matrix_pointer - (char*)weight_buf.data_ptr();
        TORCH_INTERNAL_ASSERT(
            offset_bytes % elem_size == 0,
            "offset_bytes = ",
            offset_bytes,
            "; elem_size = ",
            elem_size);
        size_t offset = offset_bytes / elem_size;
        // for all the RNN types provided by CUDNN, all the ih weights
        // are the same size and are allocated in a contiguous chunk
        // (same for the hh weights, and the ih and hh biases).
        // Since we're storing all the weights in a single tensor anyway,
        // might as well merge the CUDNN ones into a single tensor as well
        int mat_numel =
            c10::multiply_integers(filter_dim_a, filter_dim_a + nb_dims);
        if (linear_id == 0 || linear_id == num_linear_layers / 2) {
          // We could also exclude bias params by restricting cudnn_methods to
          // just { cudnnGetRNNLinLayerMatrixParams } at the very top.  However,
          // to do so would throw off the cur_offset account, which is currently
          // a strict and informative check that all params are laid out the way
          // we think they are.  If include_bias is false, I'd rather keep full
          // cur_offset checks rather than save some CPU overhead by skipping
          // the cudnn_method = cudnnGetRNNLinLayerBiasParams iteration.
#ifndef USE_CUDNN_RNN_V8_API
          if (include_bias || cudnn_method != cudnnGetRNNLinLayerBiasParams) {
#else
          if (include_bias || cudnn_method) {
#endif
            // Generate a new parameter tensor which is a view into the
            // weight_buf.
            std::initializer_list<int64_t> size = {
                mat_numel * num_linear_layers / 2, 1};
            Tensor param = at::empty({0}, weight_buf.options())
                               .set_(weight_buf.storage(), offset, size);
            params.emplace_back(std::move(param));
            layer_params_count++;
          }
        } else {
          TORCH_INTERNAL_ASSERT(
              cur_offset == offset,
              "cur_offset = ",
              cur_offset,
              "; offset = ",
              offset);
        }
        cur_offset = offset + mat_numel;
      }
    } // for cudnn_method
    if (rnn.proj_size != 0) {
#ifndef USE_CUDNN_RNN_V8_API
      add_projection_weights(
          handle, rnn_desc, x_desc, w_desc, weight_buf, layer, params);
#else
      add_projection_weights(handle, rnn_desc, weight_buf, layer, params);
#endif
      layer_params_count++;
    }

    if (layer == 0) {
      global_layer_params_count = layer_params_count;
    } else {
      TORCH_INTERNAL_ASSERT(
          global_layer_params_count == layer_params_count,
          "global_layer_params_count = ",
          global_layer_params_count,
          "; layer_params_count = ",
          layer_params_count);
    }
  } // for layer
  return std::make_pair(params, global_layer_params_count);
}

// This is a lightweight version of the method above used to quickly get the
// expected parameter offsets.
std::vector<void*> get_expected_data_ptrs(
    const Tensor& weight_buf,
    cudnnHandle_t handle,
    const RNNDescriptorParams& rnn,
    const RNNDescriptor& rnn_desc,
    const TensorDescriptor& x_desc,
    cudnnDataType_t datatype) {
#ifndef USE_CUDNN_RNN_V8_API
  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);
#endif

  int64_t num_linear_layers = _num_linear_layers(rnn.mode);
  int64_t num_dir_layers = rnn.num_directions() * rnn.num_layers;
#ifndef USE_CUDNN_RNN_V8_API
  const auto cudnn_methods = {
      cudnnGetRNNLinLayerMatrixParams, cudnnGetRNNLinLayerBiasParams};
#else
  const auto cudnn_methods = {true, false};
#endif
  std::vector<void*> data_ptrs;
  if (rnn.proj_size != 0) {
    data_ptrs.reserve(num_dir_layers * (2 * 2 + 1));
  } else {
    data_ptrs.reserve(num_dir_layers * 2 * 2);
  }
  for (const auto layer : c10::irange(num_dir_layers)) {
    for (auto cudnn_method : cudnn_methods) {
      // This API returns a separate pointer for weight of every gate,
      // but we represent them as a single tensor, so we're only interested
      // in a very limited subset of possible values.
      const std::array<int64_t, 2> linear_offsets = {0, num_linear_layers / 2};
      for (int64_t linear_id : linear_offsets) {
        void* matrix_pointer;
#ifndef USE_CUDNN_RNN_V8_API
        FilterDescriptor lin_layer_mat_desc;
        AT_CUDNN_CHECK(cudnn_method(
            handle,
            rnn_desc.desc(),
            layer,
            x_desc.desc(),
            w_desc.desc(),
            weight_buf.data_ptr(),
            linear_id,
            lin_layer_mat_desc.mut_desc(),
            &matrix_pointer));
#else
        void* unused_pointer = nullptr;
        TensorDescriptor unused_desc;
        TensorDescriptor lin_layer_mat_desc;
        if (cudnn_method) { // matrix
          AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
              handle,
              rnn_desc.desc(),
              layer,
              weight_buf.numel() * weight_buf.element_size(),
              weight_buf.data_ptr(),
              linear_id,
              lin_layer_mat_desc.mut_desc(),
              &matrix_pointer,
              unused_desc.mut_desc(),
              &unused_pointer));
        } else { // bias
          AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
              handle,
              rnn_desc.desc(),
              layer,
              weight_buf.numel() * weight_buf.element_size(),
              weight_buf.data_ptr(),
              linear_id,
              unused_desc.mut_desc(),
              &unused_pointer,
              lin_layer_mat_desc.mut_desc(),
              &matrix_pointer));
        }
#endif
        data_ptrs.push_back(matrix_pointer);
      }
    }
    if (rnn.proj_size != 0) {
      // assuming it's LSTM which has 8 "linear layers" (i.e. 4 weights and 4
      // biases)
      int64_t linear_id = 8;
      void* matrix_pointer;
#ifndef USE_CUDNN_RNN_V8_API
      FilterDescriptor lin_layer_mat_desc;
      AT_CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(
          handle,
          rnn_desc.desc(),
          layer,
          x_desc.desc(),
          w_desc.desc(),
          weight_buf.data_ptr(),
          linear_id,
          lin_layer_mat_desc.mut_desc(),
          &matrix_pointer));
#else
      void* unused_pointer;
      TensorDescriptor unused_desc;
      TensorDescriptor lin_layer_mat_desc;

      AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
          handle,
          rnn_desc.desc(),
          layer,
          weight_buf.numel() * weight_buf.element_size(),
          weight_buf.data_ptr(),
          linear_id,
          lin_layer_mat_desc.mut_desc(),
          &matrix_pointer,
          unused_desc.mut_desc(),
          &unused_pointer));
#endif
      data_ptrs.push_back(matrix_pointer);
    }
  }
  return data_ptrs;
}

void _viewOrCopyOneParam(
    const Tensor& param_from,
    const Tensor& param_to,
    bool copy,
    bool allow_type_change = false) {
  // if copying, allow_type_change may be true or false.
  // if viewing, allow_type_change must be false.
  TORCH_INTERNAL_ASSERT(
      copy || !allow_type_change, "if viewing, type change is not allowed.");
  TORCH_INTERNAL_ASSERT(
      allow_type_change || (param_from.scalar_type() == param_to.scalar_type()),
      "parameter types mismatch");
  if (copy) {
    param_to.copy_(param_from.view_as(param_to));
  } else {
    param_from.resize_as_(param_to);
  }
}

void _viewOrCopyParams(
    MatrixRef<Tensor> params_from,
    MatrixRef<Tensor> params_to,
    bool copy,
    bool allow_type_change = false) {
  TORCH_INTERNAL_ASSERT(
      params_from.size(0) == params_to.size(0), "number of layers mismatch");
  for (const auto i : c10::irange(params_from.size(0))) {
    auto layer_params_from = params_from[i];
    auto layer_params_to = params_to[i];
    // NOTE: these lists have all weights before all biases, so if the layer
    // doesn't use biases, iteration will terminate once layer_params_from ends
    // and ignore them.

    // NOTE: there is an exception from the above statement. If LSTMs with
    // projections are used, weights layout will be w_ih, w_hh, b_ih, b_hh,
    // w_hr. So need to handle no-bias case specially, because will need to copy
    // 0->0, 1->1, 2->4. This case can be uniquely identified by checking if
    // number of defined parameters for each layer is 3.
    if (layer_params_from.size() == 3 && layer_params_to.size() != 3) {
      _viewOrCopyOneParam(
          layer_params_from[0], layer_params_to[0], copy, allow_type_change);
      _viewOrCopyOneParam(
          layer_params_from[1], layer_params_to[1], copy, allow_type_change);
      _viewOrCopyOneParam(
          layer_params_from[2], layer_params_to[4], copy, allow_type_change);
      continue;
    }
    if (layer_params_to.size() == 3 && layer_params_from.size() != 3) {
      _viewOrCopyOneParam(
          layer_params_from[0], layer_params_to[0], copy, allow_type_change);
      _viewOrCopyOneParam(
          layer_params_from[1], layer_params_to[1], copy, allow_type_change);
      _viewOrCopyOneParam(
          layer_params_from[4], layer_params_to[2], copy, allow_type_change);
      continue;
    }
    for (auto a = layer_params_from.begin(), b = layer_params_to.begin();
         a != layer_params_from.end() && b != layer_params_to.end();
         ++a, ++b) {
      _viewOrCopyOneParam(*a, *b, copy, allow_type_change);
    }
  }
}

void _copyParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
  _viewOrCopyParams(params_from, params_to, true);
}

void _viewParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
  _viewOrCopyParams(params_from, params_to, false);
}

std::vector<int64_t> _input_size(const TensorDescriptorListParams& tensors) {
  if (tensors.is_input_packed()) {
    return {tensors.batch_sizes_sum, tensors.input_size};
  } else {
    return {tensors.seq_length, tensors.mini_batch, tensors.input_size};
  }
}

std::vector<int64_t> _hidden_size(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  if (rnn.proj_size != 0) {
    return {
        rnn.num_layers * rnn.num_directions(),
        tensors.mini_batch,
        rnn.proj_size};
  } else {
    return {
        rnn.num_layers * rnn.num_directions(),
        tensors.mini_batch,
        rnn.hidden_size};
  }
}

std::vector<int64_t> _cell_size(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  return {
      rnn.num_layers * rnn.num_directions(),
      tensors.mini_batch,
      rnn.hidden_size};
}

std::vector<int64_t> _output_size(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  auto out_size = rnn.hidden_size;
  if (rnn.proj_size != 0) {
    out_size = rnn.proj_size;
  }
  if (tensors.is_input_packed()) {
    return {tensors.batch_sizes_sum, out_size * rnn.num_directions()};
  } else {
    return {
        tensors.seq_length,
        tensors.mini_batch,
        out_size * rnn.num_directions()};
  }
}

inline bool use_persist_common_heuristics(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  return rnn.num_layers == 1 && rnn.hidden_size <= 1024 &&
      rnn.num_directions() == 1 && rnn.hidden_size % 128 == 0 &&
      tensors.input_size % 128 == 0;
}

inline bool use_persist_device_heuristics(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  auto bsize = tensors.mini_batch;
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major == 7) {
    if (prop->minor == 5) {
      // Excludes Turing from using persistent rnn.
      return false;
    } else {
      // technically, batch size should be multiple of 8, but there are quite a
      // few multiple-of-8 batchsizes that give bad perf, weed them out
      return ((bsize % 16 == 0 && bsize != 80 && bsize != 112) || bsize == 8) &&
          ((tensors.seq_length >= 40 && bsize <= 128) ||
           (tensors.seq_length >= 20 && bsize <= 96) ||
           (tensors.seq_length >= 10 && bsize <= 32));
    }
  } else if (prop->major >= 8 && prop->multiProcessorCount >= 98) {
    // SM count check excludes A30 (similar issue to A40)
    if (prop->minor == 6) {
      // Excludes sm_86 GPU devices from using persistent rnn.
      // This is because there are some edge cases that will throw exceptions
      // with cudnn 8.0.5 on Nvidia A40 GPU.
      return false;
    }
    // Based on tests by Vasily Volkov and xwang233.  Vasily only tried bsize <=
    // 128, so conservatively enable persistence for bsize <= 128 only.
    // TODO:  Run more tests for bsize > 128.
    if (rnn.mode == CUDNN_GRU) {
      // Persistent GRU performance is flakier than other RNN types.  Exclude
      // them for now.
      // TODO:  Write a more refined GRU heuristic.
      return false;
    } else if (rnn.mode == CUDNN_LSTM) {
      // Persistent LSTMs are comparable to or better than non-persistent for
      // bsize <= 128.
      return (bsize % 8 == 0) && (bsize <= 128);
    } else {
      // Persistent RNN_RELU and TANH show poor performance when bsize >= 96 AND
      // hidden size >= 896.
      return (bsize % 8 == 0) && (bsize <= 128) &&
          (bsize < 96 || rnn.hidden_size < 896);
    }
  } else {
    return false;
  }
}

inline bool use_rnn_persist_small_h(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors,
    bool forward) {
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8201 // 8.2.1
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major < 6)
    return false;

  if (forward) {
    if (rnn.mode == CUDNN_RNN_RELU || rnn.mode == CUDNN_RNN_TANH) {
      return rnn.hidden_size <= 384;
    }
    if (rnn.mode == CUDNN_LSTM || rnn.mode == CUDNN_GRU) {
      return rnn.hidden_size <= 192;
    }
  } else /* backward */ {
    if (rnn.mode == CUDNN_RNN_RELU || rnn.mode == CUDNN_RNN_TANH) {
      return rnn.hidden_size <= 256;
    }
    if (rnn.mode == CUDNN_LSTM || rnn.mode == CUDNN_GRU) {
      return rnn.hidden_size <= 128;
    }
  }

  return false;
#else
  return false;
#endif
}

cudnnRNNAlgo_t get_algo(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors,
    const Tensor input,
    bool forward) {
  // LSTM with projections only works with standard algorithm
  if (rnn.proj_size != 0) {
    return CUDNN_RNN_ALGO_STANDARD;
  }

  // Persistent algos typically don't work for packed inputs with sequence
  // lengths that vary across batch elements, and will return
  // CUDNN_STATUS_NOT_SUPPORTED if attempted. See
  // https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#features-of-rnn-functions
  if (!tensors.is_input_packed()) {
    auto cudnnDataType = getCudnnDataType(input);
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8201 // 8.2.1
    if (cudnnDataType != CUDNN_DATA_DOUBLE) {
      if (use_rnn_persist_small_h(rnn, tensors, forward)) {
        return CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H;
      }
    }
#endif
    if (cudnnDataType == CUDNN_DATA_HALF) {
      if (use_persist_common_heuristics(rnn, tensors) &&
          use_persist_device_heuristics(rnn, tensors)) {
        return CUDNN_RNN_ALGO_PERSIST_STATIC;
      }
    }
  }

  return CUDNN_RNN_ALGO_STANDARD;
}

cudnnDataType_t promote_rnn_math_type(cudnnDataType_t dtype) {
  if (dtype == CUDNN_DATA_HALF) {
    return CUDNN_DATA_FLOAT;
  }
  return dtype;
}

} // namespace native

// Utilities exposed in RNNUtils.h
namespace cudnn_rnn {

TORCH_CUDA_CPP_API std::tuple<Tensor, std::vector<Tensor>>
copy_weights_to_flat_buf_views(
    TensorList weight_arr,
    int64_t weight_stride0,
    int64_t input_size,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    bool bidirectional,
    const cudnnDataType_t flat_buf_datatype,
    const TensorOptions& flat_buf_options,
    bool set_orig_weights_to_flat_buf,
    bool allow_type_change /*=false*/,
    bool include_bias /*=true*/) {
  // flat_buf_datatype is accepted as a separate argument (rather than extracted
  // from flat_buf_options) because to extract flat_buf_datatype from
  // flat_buf_options, we'd need to say auto flat_buf_datatype =
  // getCudnnDataTypeFromScalarType(typeMetaToScalarType(options.dtype()));
  // typeMetaToScalarType is a surprisingly nontrivial function.  We should
  // avoid it if we can.
  TORCH_CHECK(
      weight_arr.size() > 0,
      "copy_weights_to_flat_buf_views: cannot flatten empty weight list");

  RNNDescriptorParams rnn;
  rnn.set(
      mode,
#ifdef USE_CUDNN_RNN_V8_API
      input_size,
      false, // eqy: bogus as we do not know if the input is packed here
             // but it should not affect the weights (what are are interested
             // in)
#endif
      hidden_size,
      proj_size,
      num_layers,
      bidirectional,
      promote_rnn_math_type(flat_buf_datatype),
      flat_buf_datatype);

  auto handle = getCudnnHandle();
  RNNDescriptor rnn_desc = rnn.descriptor(handle);

  TensorGeometry x_geom({1, input_size});
  TensorDescriptor x_desc;
  // Why do we pad to 5 dims here (and elsewhere)?
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnRNNForwardTraining
  // expects descriptors padded to 3 dimensions.
  x_desc.set(flat_buf_datatype, x_geom.sizes(), x_geom.strides(), 5);

  auto num_weights =
#ifndef USE_CUDNN_RNN_V8_API
      get_num_weights(handle, rnn_desc, x_desc, flat_buf_datatype);
#else
      get_num_weights(handle, rnn_desc, flat_buf_datatype);
#endif
  auto weight_buf = at::zeros(num_weights, flat_buf_options);

#ifndef USE_CUDNN_RNN_V8_API
  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);
#endif

  // Slice off views into weight_buf
  auto [params_arr, params_stride0] = get_parameters(
#ifndef USE_CUDNN_RNN_V8_API
      handle, rnn, rnn_desc, x_desc, w_desc, weight_buf, include_bias);
#else
      handle, rnn, rnn_desc, weight_buf, include_bias);
#endif
  MatrixRef<Tensor> weight{weight_arr, static_cast<size_t>(weight_stride0)},
      params{params_arr, params_stride0};

  // Copy weights
  _viewOrCopyParams(weight, params, /*copy=*/true, allow_type_change);
  if (set_orig_weights_to_flat_buf) {
    // Update the storage
    for (const auto i : c10::irange(weight.size(0))) {
      // There is a special case for LSTM with projections and no bias,
      // where weight copy is done in 0->0, 1->1, 2->4 layout
      if (weight[i].size() == 3 && params[i].size() == 5) {
        weight[i][0].set_(params[i][0].view_as(weight[i][0]));
        weight[i][1].set_(params[i][1].view_as(weight[i][1]));
        weight[i][2].set_(params[i][4].view_as(weight[i][2]));
      } else {
        for (auto orig_param_it = weight[i].begin(),
                  new_param_it = params[i].begin();
             orig_param_it != weight[i].end() &&
             new_param_it != params[i].end();
             orig_param_it++, new_param_it++) {
          auto orig_param = *orig_param_it, new_param = *new_param_it;
          orig_param.set_(new_param.view_as(orig_param));
        }
      }
    }
  }

  return std::make_tuple(weight_buf, params_arr);
}

} // namespace cudnn_rnn

using namespace cudnn_rnn;

// NB: does inplace update into TensorList
// It would be a relatively simple matter to refactor this into multiple
// functions, only one of which does an inplace update, but we leave this
// for future work
Tensor _cudnn_rnn_flatten_weight(
    TensorList weight_arr,
    int64_t weight_stride0,
    int64_t input_size,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    bool fn_bidirectional) {
  // returns flat weight_buf
  return std::get<0>(copy_weights_to_flat_buf_views(
      weight_arr,
      weight_stride0,
      input_size,
      fn_mode,
      fn_hidden_size,
      fn_proj_size,
      fn_num_layers,
      batch_first,
      fn_bidirectional,
      /*flat_buf_datatype=*/getCudnnDataType(weight_arr[0]),
      /*flat_buf_options=*/weight_arr[0].options(),
      /*set_orig_weights_to_flat_buf=*/true));
}

const char* WEIGHT_FORMAT_WARN =
    "RNN module weights are not part of single contiguous "
    "chunk of memory. This means they need to be compacted "
    "at every call, possibly greatly increasing memory usage. "
    "To compact weights again call flatten_parameters().";

// NB: when fn_batch_sizes is empty, that means no batch sizes was specified
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _cudnn_rnn(
    const Tensor& input_r,
    TensorList weight,
    int64_t weight_stride0,
    const c10::optional<Tensor>& weight_buf_r_opt,
    const Tensor& hx,
    const c10::optional<Tensor>& cx_opt,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    double fn_dropout,
    bool fn_train,
    bool fn_bidirectional,
    IntArrayRef fn_batch_sizes,
    const c10::optional<Tensor>& fn_dropout_state_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_buf_r_maybe_owned =
      at::borrow_from_optional_tensor(weight_buf_r_opt);
  const Tensor& weight_buf_r = *weight_buf_r_maybe_owned;
  const Tensor& cx = c10::value_or_else(cx_opt, [] { return Tensor(); });
  const Tensor& fn_dropout_state =
      c10::value_or_else(fn_dropout_state_opt, [] { return Tensor(); });

  check_attributes(input_r, weight, {hx, cx}, /*check_dtype=*/true);
  auto input = input_r;
  auto weight_buf = weight_buf_r;
  if (!weight_buf.defined()) {
    TORCH_WARN(WEIGHT_FORMAT_WARN);
  }
  if (fn_dropout_state.defined()) {
    auto input_arg = TensorArg(input, "input", 1);
    auto dropout_state_arg = TensorArg(fn_dropout_state, "dropout_states", 15);
    checkSameGPU("cudnn_rnn", input_arg, dropout_state_arg);
  }
  RNNParams fn;
  auto datatype = getCudnnDataType(input);
#ifndef USE_CUDNN_RNN_V8_API
  fn.rnn.set(
      fn_mode,
      fn_hidden_size,
      fn_proj_size,
      fn_num_layers,
      fn_bidirectional,
      promote_rnn_math_type(datatype),
      datatype);
#else
  auto input_size = input_r.size(-1);
  auto packed = fn_batch_sizes.size() != 0;
  fn.rnn.set(
      fn_mode,
      input_size,
      packed,
      fn_hidden_size,
      fn_proj_size,
      fn_num_layers,
      fn_bidirectional,
      promote_rnn_math_type(datatype),
      datatype);
#endif
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  // TODO: Set device to input

  if (fn.rnn.mode != CUDNN_LSTM) {
    TORCH_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
  }

  // TODO: can batch_first be a wrapper around this function?
  auto is_input_packed = fn.tensors.batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
  }

  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  auto cell_size = _cell_size(fn.rnn, fn.tensors);
  auto output_size = _output_size(fn.rnn, fn.tensors);

  TORCH_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
  TORCH_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");

  auto x = input.contiguous();
  auto output = at::empty(output_size, input.options());
  auto hy = at::empty(hidden_size, hx.options());
  Tensor cy;
  if (cx.defined()) {
    cy = at::empty(cell_size, cx.options());
  } else {
    cy = at::empty(
        {0}, hx.options()); // NB: Not allowed to return undefined tensors
  }
  auto y = output;

  auto handle = getCudnnHandle();
  cudnnRNNAlgo_t algo = get_algo(fn.rnn, fn.tensors, input, true);
  fn.rnn.set_algo(algo);
  RNNDescriptors descs(fn, handle, x, y, hx, cx);

#ifndef USE_CUDNN_RNN_V8_API
  FilterDescriptor w_desc;
#endif
  if (!weight_buf.defined()) {
#ifndef USE_CUDNN_RNN_V8_API
    auto num_weights =
        get_num_weights(handle, descs.rnn_desc, descs.x_descs[0], datatype);
#else
    auto num_weights = get_num_weights(handle, descs.rnn_desc, datatype);
#endif
    weight_buf = at::empty(num_weights, x.options());
#ifndef USE_CUDNN_RNN_V8_API
    w_desc.set(weight_buf, 3);
#endif
    weight_buf.zero_();
#ifndef USE_CUDNN_RNN_V8_API
    auto [params, params_stride0] = get_parameters(
        handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, weight_buf);
#else
    auto [params, params_stride0] =
        get_parameters(handle, fn.rnn, descs.rnn_desc, weight_buf);
#endif
    _copyParams(
        MatrixRef<Tensor>{weight, static_cast<size_t>(weight_stride0)},
        MatrixRef<Tensor>{params, params_stride0});
  } else {
#ifndef USE_CUDNN_RNN_V8_API
    w_desc.set(weight_buf, 3);
#endif
  }

  TORCH_CHECK(
      !cx.defined() || cx.sizes().equals(cell_size),
      "Expected cell size ",
      IntArrayRef{cell_size},
      ", got ",
      cx.sizes());
  size_t workspace_size;
#ifndef USE_CUDNN_RNN_V8_API
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
#else
  auto& x_descs_arr = descs.x_descs;
  auto& y_descs_arr = descs.y_descs;
#endif
#ifndef USE_CUDNN_RNN_V8_API
  AT_CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
      handle,
      descs.rnn_desc.desc(),
      fn.tensors.seq_length,
      x_descs_arr.data(),
      &workspace_size));
#endif
  Tensor workspace;
  Tensor reserve;
  // NB: Previously, the test was for fn.requires_grad, but we don't have
  // this information.  Use 'train' as a proxy.
  if (fn_train) {
    size_t reserve_size;
#ifndef USE_CUDNN_RNN_V8_API
    AT_CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        &reserve_size));
#else
    AT_CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(
        handle,
        descs.rnn_desc.desc(),
        CUDNN_FWD_MODE_TRAINING,
        x_descs_arr.desc(),
        &workspace_size,
        &reserve_size));
#endif
    workspace = at::empty(workspace_size, input.options().dtype(kByte));
    reserve = at::empty(reserve_size, input.options().dtype(kByte));
#ifndef USE_CUDNN_RNN_V8_API
    AT_CUDNN_CHECK(cudnnRNNForwardTraining(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        x.data_ptr(),
        descs.hx_desc.desc(),
        hx.data_ptr(),
        descs.cx_desc.desc(),
        cx.defined() ? cx.data_ptr() : nullptr,
        w_desc.desc(),
        weight_buf.data_ptr(),
        y_descs_arr.data(),
        y.data_ptr(),
        descs.hy_desc.desc(),
        hy.data_ptr(),
        descs.cy_desc.desc(),
        cy.defined() ? cy.data_ptr() : nullptr,
        workspace.data_ptr(),
        workspace.size(0),
        reserve.mutable_data_ptr(),
        reserve.size(0)));
#else
    AT_CUDNN_CHECK(cudnnRNNForward(
        handle,
        descs.rnn_desc.desc(),
        CUDNN_FWD_MODE_TRAINING,
        nullptr,
        x_descs_arr.desc(),
        x.data_ptr(),
        y_descs_arr.desc(),
        y.data_ptr(),
        descs.hx_desc.desc(),
        hx.data_ptr(),
        hy.data_ptr(),
        descs.cx_desc.desc(),
        cx.defined() ? cx.data_ptr() : nullptr,
        cy.defined() ? cy.data_ptr() : nullptr,
        weight_buf.numel() * weight_buf.element_size(),
        weight_buf.data_ptr(),
        workspace.size(0),
        workspace.data_ptr(),
        reserve.size(0),
        reserve.mutable_data_ptr()));
#endif
  } else { // inference
#ifdef USE_CUDNN_RNN_V8_API
    AT_CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(
        handle,
        descs.rnn_desc.desc(),
        CUDNN_FWD_MODE_INFERENCE,
        x_descs_arr.desc(),
        &workspace_size,
        NULL));
#endif
    workspace = at::empty(workspace_size, input.options().dtype(kByte));
    reserve = at::empty({0}, input.options().dtype(kByte));
#ifndef USE_CUDNN_RNN_V8_API
    AT_CUDNN_CHECK(cudnnRNNForwardInference(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        x.data_ptr(),
        descs.hx_desc.desc(),
        hx.data_ptr(),
        descs.cx_desc.desc(),
        cx.defined() ? cx.data_ptr() : nullptr,
        w_desc.desc(),
        weight_buf.data_ptr(),
        y_descs_arr.data(),
        y.data_ptr(),
        descs.hy_desc.desc(),
        hy.data_ptr(),
        descs.cy_desc.desc(),
        cy.defined() ? cy.data_ptr() : nullptr,
        workspace.data_ptr(),
        workspace.size(0)));
#else
    AT_CUDNN_CHECK(cudnnRNNForward(
        handle,
        descs.rnn_desc.desc(),
        CUDNN_FWD_MODE_INFERENCE,
        nullptr,
        x_descs_arr.desc(),
        x.data_ptr(),
        y_descs_arr.desc(),
        y.data_ptr(),
        descs.hx_desc.desc(),
        hx.data_ptr(),
        hy.data_ptr(),
        descs.cx_desc.desc(),
        cx.defined() ? cx.data_ptr() : nullptr,
        cy.defined() ? cy.data_ptr() : nullptr,
        weight_buf.numel() * weight_buf.element_size(),
        weight_buf.data_ptr(),
        workspace.size(0),
        workspace.data_ptr(),
        reserve.size(0),
        reserve.mutable_data_ptr()));
#endif
  }

  if (batch_first && !is_input_packed) {
    output.transpose_(0, 1);
  }

  return std::make_tuple(output, hy, cy, reserve, weight_buf);
}

std::tuple<Tensor, Tensor, Tensor> _cudnn_rnn_backward_input(
    const Tensor& input_r,
    const Tensor& weight_buf,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& output_r,
    const Tensor& grad_output_r,
    const Tensor& grad_hy,
    const Tensor& grad_cy,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    double fn_dropout,
    bool fn_train,
    bool fn_bidirectional,
    IntArrayRef fn_batch_sizes,
    const Tensor& fn_dropout_state,
    const Tensor& fn_reserve,
    std::array<bool, 3> output_mask) {
  auto input = input_r;
  auto grad_output = grad_output_r;
  auto output = output_r;

  RNNParams fn;
  auto datatype = getCudnnDataType(input);
#ifndef USE_CUDNN_RNN_V8_API
  fn.rnn.set(
      fn_mode,
      fn_hidden_size,
      fn_proj_size,
      fn_num_layers,
      fn_bidirectional,
      promote_rnn_math_type(datatype),
      datatype);
#else
  auto cudnn_input_size = input_r.size(-1);
  auto packed = fn_batch_sizes.size() != 0;
  fn.rnn.set(
      fn_mode,
      cudnn_input_size,
      packed,
      fn_hidden_size,
      fn_proj_size,
      fn_num_layers,
      fn_bidirectional,
      promote_rnn_math_type(datatype),
      datatype);
#endif
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  // TODO: Set device to input
  auto handle = getCudnnHandle();

  if (fn.rnn.mode != CUDNN_LSTM) {
    TORCH_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
  }

  auto is_input_packed = fn_batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
    grad_output = grad_output.transpose(0, 1);
    output = output.transpose(0, 1);
  }

  auto input_size = _input_size(fn.tensors);
  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  auto cell_size = _cell_size(fn.rnn, fn.tensors);
  auto output_size = _output_size(fn.rnn, fn.tensors);

  TORCH_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
  TORCH_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");

  auto x = input.contiguous();
  auto dy = grad_output.contiguous();
  auto y = output;
  auto w = weight_buf;
  auto dx = at::empty(
      input.sizes(), input.options()); // TODO: more compact way of saying this
  auto dhy = grad_hy.contiguous().view(hidden_size);
  auto dcy =
      grad_cy.defined() ? grad_cy.contiguous().view(cell_size) : Tensor();
  auto dhx = at::empty(hidden_size, hx.options());
  TORCH_INTERNAL_ASSERT(
      cx.defined() || !output_mask[2],
      "illegally required grad of cx for non-LSTM RNN");
  auto dcx = cx.defined() ? at::empty(cell_size, cx.options()) : Tensor();

  TORCH_CHECK(
      fn_train, "cudnn RNN backward can only be called in training mode");

  TORCH_CHECK(
      input.sizes().equals(input_size),
      "Expected input size ",
      IntArrayRef{input_size},
      ", got ",
      input.sizes());
  TORCH_CHECK(
      output.sizes().equals(output_size),
      "Expected output size ",
      IntArrayRef{output_size},
      ", got ",
      output.sizes());

  TORCH_CHECK(
      !hx.defined() || hx.sizes().equals(hidden_size),
      "Expected hidden size ",
      IntArrayRef{hidden_size},
      ", got ",
      hx.sizes());
  TORCH_CHECK(
      !cx.defined() || cx.sizes().equals(cell_size),
      "Expected cell size ",
      IntArrayRef{cell_size},
      ", got ",
      cx.sizes());
  TORCH_CHECK(
      !dhy.defined() || dhy.sizes().equals(hidden_size),
      "Expected d_hidden size ",
      IntArrayRef{hidden_size},
      ", got ",
      dhy.sizes());
  TORCH_CHECK(
      !dcy.defined() || dcy.sizes().equals(cell_size),
      "Expected d_cell size ",
      IntArrayRef{cell_size},
      ", got ",
      dcy.sizes());

  TORCH_CHECK(
      dhy.is_cuda() && dy.is_cuda() && (!dcy.defined() || dcy.is_cuda()),
      "Gradients aren't CUDA tensors");

  cudnnRNNAlgo_t algo = get_algo(fn.rnn, fn.tensors, input, false);
  fn.rnn.set_algo(algo);
  RNNDescriptors descs(fn, handle, x, y, hx, cx);

#ifndef USE_CUDNN_RNN_V8_API
  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);
#endif

  size_t workspace_size;
#ifndef USE_CUDNN_RNN_V8_API
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  AT_CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
      handle,
      descs.rnn_desc.desc(),
      fn.tensors.seq_length,
      x_descs_arr.data(),
      &workspace_size));
#else
  auto& x_descs_arr = descs.x_descs;
  auto& y_descs_arr = descs.y_descs;
  AT_CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(
      handle,
      descs.rnn_desc.desc(),
      CUDNN_FWD_MODE_TRAINING,
      x_descs_arr.desc(),
      &workspace_size,
      NULL));
#endif
  // TODO: put this in the correct device???
  Tensor workspace = at::empty(workspace_size, input.options().dtype(kByte));
#ifndef USE_CUDNN_RNN_V8_API
  AT_CUDNN_CHECK(cudnnRNNBackwardData(
      handle,
      descs.rnn_desc.desc(),
      fn.tensors.seq_length,
      y_descs_arr.data(),
      y.data_ptr(),
      y_descs_arr.data(),
      dy.data_ptr(),
      descs.hy_desc.desc(),
      dhy.data_ptr(),
      descs.cy_desc.desc(),
      cx.defined() ? dcy.data_ptr() : nullptr,
      w_desc.desc(),
      w.data_ptr(),
      descs.hx_desc.desc(),
      hx.data_ptr(),
      descs.cx_desc.desc(),
      cx.defined() ? cx.data_ptr() : nullptr,
      x_descs_arr.data(),
      dx.data_ptr(),
      descs.hx_desc.desc(),
      dhx.data_ptr(),
      descs.cx_desc.desc(),
      cx.defined() ? dcx.data_ptr() : nullptr,
      workspace.data_ptr(),
      workspace.size(0),
      fn_reserve.data_ptr(),
      fn_reserve.size(0)));
#else
  AT_CUDNN_CHECK(cudnnRNNBackwardData_v8(
      handle,
      descs.rnn_desc.desc(),
      nullptr,
      y_descs_arr.desc(),
      y.data_ptr(),
      dy.data_ptr(),
      x_descs_arr.desc(),
      dx.data_ptr(),
      descs.hx_desc.desc(),
      hx.data_ptr(),
      dhy.data_ptr(),
      dhx.data_ptr(),
      descs.cx_desc.desc(),
      cx.defined() ? cx.data_ptr() : nullptr,
      cx.defined() ? dcy.data_ptr() : nullptr,
      cx.defined() ? dcx.data_ptr() : nullptr,
      weight_buf.numel() * weight_buf.element_size(),
      weight_buf.data_ptr(),
      workspace.size(0),
      workspace.data_ptr(),
      fn_reserve.size(0),
      fn_reserve.data_ptr()));
#endif
  if (batch_first && !is_input_packed) {
    dx = dx.transpose_(0, 1);
  }

  return std::make_tuple(dx, dhx, dcx);
}

// NB: This MUST BE CALLED AFTER _cudnn_rnn_backward_input.
// We'll give a user friendly combined function...
std::vector<Tensor> _cudnn_rnn_backward_weight(
    // TODO: I think tensor geometry sufficient for weight_buf/weight
    const Tensor& input_r,
    TensorList weight_arr,
    int64_t weight_stride0,
    const Tensor& weight_buf,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& output_r,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    double fn_dropout,
    bool fn_train,
    bool fn_bidirectional,
    IntArrayRef fn_batch_sizes,
    const Tensor& fn_dropout_state,
    const Tensor& fn_reserve) {
  MatrixRef<Tensor> weight{weight_arr, static_cast<size_t>(weight_stride0)};
  auto input = input_r;
  auto output = output_r;

  RNNParams fn;
  auto datatype = getCudnnDataType(input);
#ifndef USE_CUDNN_RNN_V8_API
  fn.rnn.set(
      fn_mode,
      fn_hidden_size,
      fn_proj_size,
      fn_num_layers,
      fn_bidirectional,
      promote_rnn_math_type(datatype),
      datatype);
#else
  auto cudnn_input_size = input_r.size(-1);
  auto packed = fn_batch_sizes.size() != 0;
  fn.rnn.set(
      fn_mode,
      cudnn_input_size,
      packed,
      fn_hidden_size,
      fn_proj_size,
      fn_num_layers,
      fn_bidirectional,
      promote_rnn_math_type(datatype),
      datatype);
#endif
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  auto handle = getCudnnHandle();

  if (fn.rnn.mode != CUDNN_LSTM) {
    TORCH_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
  }

  auto is_input_packed = fn_batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
    output = output.transpose(0, 1);
  }

  auto input_size = _input_size(fn.tensors);
  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);

  TORCH_CHECK(
      fn_train, "cudnn RNN backward can only be called in training mode");

  TORCH_CHECK(
      input.sizes().equals(input_size),
      "Expected input size ",
      IntArrayRef{input_size},
      ", got ",
      input.sizes());
  TORCH_CHECK(
      !hx.defined() || hx.sizes().equals(hidden_size),
      "Expected hidden size ",
      IntArrayRef{hidden_size},
      ", got ",
      hx.sizes());

  // TODO: the above were the only checks in rnn.py, but it doesn't seem
  // like these checks are enough

  TORCH_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
  TORCH_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");

  auto x = input.contiguous();
  const auto& y = output;
  auto dw = at::zeros(weight_buf.sizes(), weight_buf.options());

  cudnnRNNAlgo_t algo = get_algo(fn.rnn, fn.tensors, input, false);
  fn.rnn.set_algo(algo);
  RNNDescriptors descs(fn, handle, x, y, hx, cx);

#ifndef USE_CUDNN_RNN_V8_API
  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);
#endif

  size_t workspace_size;
#ifndef USE_CUDNN_RNN_V8_API
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  AT_CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
      handle,
      descs.rnn_desc.desc(),
      fn.tensors.seq_length,
      x_descs_arr.data(),
      &workspace_size));
#else
  auto& x_descs_arr = descs.x_descs;
  auto& y_descs_arr = descs.y_descs;
  AT_CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(
      handle,
      descs.rnn_desc.desc(),
      CUDNN_FWD_MODE_TRAINING,
      x_descs_arr.desc(),
      &workspace_size,
      NULL));
#endif
  Tensor workspace = at::empty(workspace_size, input.options().dtype(kByte));
#ifndef USE_CUDNN_RNN_V8_API
  AT_CUDNN_CHECK(cudnnRNNBackwardWeights(
      handle,
      descs.rnn_desc.desc(),
      fn.tensors.seq_length,
      x_descs_arr.data(),
      x.data_ptr(),
      descs.hx_desc.desc(),
      hx.data_ptr(),
      y_descs_arr.data(),
      y.data_ptr(),
      workspace.data_ptr(),
      workspace.size(0),
      w_desc.desc(),
      dw.data_ptr(),
      fn_reserve.data_ptr(),
      fn_reserve.size(0)));
#else
  AT_CUDNN_CHECK(cudnnRNNBackwardWeights_v8(
      handle,
      descs.rnn_desc.desc(),
      CUDNN_WGRAD_MODE_ADD,
      nullptr,
      x_descs_arr.desc(),
      x.data_ptr(),
      descs.hx_desc.desc(),
      hx.data_ptr(),
      y_descs_arr.desc(),
      y.data_ptr(),
      weight_buf.numel() * weight_buf.element_size(),
      dw.data_ptr(),
      workspace.size(0),
      workspace.data_ptr(),
      fn_reserve.size(0),
      fn_reserve.data_ptr()));
#endif

#ifndef USE_CUDNN_RNN_V8_API
  auto [grad_params_arr, grad_params_stride0] = get_parameters(
      handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, dw);
#else
  auto [grad_params_arr, grad_params_stride0] =
      get_parameters(handle, fn.rnn, descs.rnn_desc, dw);
#endif
  if (grad_params_stride0 == static_cast<size_t>(weight_stride0)) {
    _viewParams(
        MatrixRef<Tensor>{grad_params_arr, grad_params_stride0},
        MatrixRef<Tensor>{weight_arr, static_cast<size_t>(weight_stride0)});
    return grad_params_arr;
  } else {
    std::vector<Tensor> grad_weight_arr;
    grad_weight_arr.reserve(weight.numel());
    for (const auto& w : weight_arr) {
      grad_weight_arr.emplace_back(at::empty(w.sizes(), w.options()));
    }
    _copyParams(
        MatrixRef<Tensor>{grad_params_arr, grad_params_stride0},
        MatrixRef<Tensor>{
            grad_weight_arr, static_cast<size_t>(weight_stride0)});
    return grad_weight_arr;
  }
}

// We need this dispatcher because _cudnn_rnn_backward_weight has a stringent
// ordering requirement with _cudnn_rnn_backward_input
std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _cudnn_rnn_backward(
    const Tensor& input,
    TensorList weight,
    int64_t weight_stride0,
    const Tensor& weight_buf,
    const Tensor& hx,
    const c10::optional<Tensor>& cx_opt,
    const Tensor& output,
    const c10::optional<Tensor>& grad_output_r_opt,
    const c10::optional<Tensor>& grad_hy_r_opt,
    const c10::optional<Tensor>& grad_cy_r_opt,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    double dropout,
    bool train,
    bool bidirectional,
    IntArrayRef batch_sizes,
    const c10::optional<Tensor>& dropout_state_opt,
    const Tensor& reserve,
    std::array<bool, 4> output_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> cx_maybe_owned =
      at::borrow_from_optional_tensor(cx_opt);
  const Tensor& cx = *cx_maybe_owned;
  const Tensor& grad_output_r =
      c10::value_or_else(grad_output_r_opt, [] { return Tensor(); });
  const Tensor& grad_hy_r =
      c10::value_or_else(grad_hy_r_opt, [] { return Tensor(); });
  const Tensor& grad_cy_r =
      c10::value_or_else(grad_cy_r_opt, [] { return Tensor(); });
  const Tensor& dropout_state =
      c10::value_or_else(dropout_state_opt, [] { return Tensor(); });

  if (!grad_output_r.defined() && !grad_hy_r.defined() &&
      !grad_cy_r.defined()) {
    return std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>>(
        Tensor(), Tensor(), Tensor(), std::vector<Tensor>(weight.size()));
  }

  auto grad_output = grad_output_r.defined()
      ? grad_output_r
      : at::zeros_like(output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_hy = grad_hy_r.defined()
      ? grad_hy_r
      : at::zeros_like(hx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_cy = cx.defined()
      ? (grad_cy_r.defined()
             ? grad_cy_r
             : at::zeros_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT))
      : grad_cy_r;

  // NB: unconditionally compute this gradient, because it mutates reserve
  auto [dx, dhx, dcx] = at::native::_cudnn_rnn_backward_input(
      input,
      weight_buf,
      hx,
      cx,
      output,
      grad_output,
      grad_hy,
      grad_cy,
      mode,
      hidden_size,
      proj_size,
      num_layers,
      batch_first,
      dropout,
      train,
      bidirectional,
      batch_sizes,
      dropout_state,
      reserve,
      {output_mask[0], output_mask[1], output_mask[2]});
  std::vector<Tensor> dw;
  if (output_mask[3]) {
    dw = at::native::_cudnn_rnn_backward_weight(
        input,
        weight,
        weight_stride0,
        weight_buf,
        hx,
        cx,
        output,
        mode,
        hidden_size,
        proj_size,
        num_layers,
        batch_first,
        dropout,
        train,
        bidirectional,
        batch_sizes,
        dropout_state,
        reserve);
  }
  return std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>>{
      dx, dhx, dcx, dw};
}

// TODO: I am not sure if we actually need the 'dropout' and 'train' parameters
// to initialize just the state tensor
//
// NB: You can have any color you like, as long as it's a CUDA byte
// tensor.  Why does this function take a TensorOptions at all in that case?
// This is a factory function: it produces tensors but takes no tensors
// as input.  The codegen currently assumes that ALL factory functions
// take TensorOptions, so it's just a lot easier for this function to
// be bound if it also does it.
Tensor _cudnn_init_dropout_state(
    double dropout,
    bool train,
    int64_t dropout_seed,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  auto handle = getCudnnHandle();
  DropoutDescriptor dropout_desc;
  auto dropout_p = train ? dropout : 0;
  dropout_desc.initialize_rng(handle, dropout_p, dropout_seed, options);
  return dropout_desc.state;
}

////////////////////////////////////////////////////////////////////////////////
// CUDA dispatch for the generic RNN ops (at::lstm, at::gru, ...)
////////////////////////////////////////////////////////////////////////////////

namespace {

// Helpers for working with different hidden types.
std::tuple<Tensor, Tensor> unpack_hidden(const Tensor& hidden) {
  return std::make_tuple(hidden, at::Tensor{});
}

std::tuple<Tensor, Tensor> unpack_hidden(
    const std::tuple<Tensor, Tensor>& hidden) {
  return hidden;
}

template <typename hidden_type>
hidden_type pack_hidden(const Tensor& hx, const Tensor& cx) {
  static_assert(
      std::is_same<hidden_type, void>::value,
      "pack_hidden not implemented for this type");
  AT_ERROR("NOT IMPLEMENTED");
}

template <>
Tensor pack_hidden<Tensor>(const Tensor& hx, const Tensor& cx) {
  AT_ASSERT(cx.numel() == 0);
  return hx;
}

template <>
std::tuple<Tensor, Tensor> pack_hidden<std::tuple<Tensor, Tensor>>(
    const Tensor& hx,
    const Tensor& cx) {
  return std::make_tuple(hx, cx);
}

/**
 * Note [DropoutState and CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * (1) Telling a capturing stream to wait on an event recorded in a
 non-capturing stream is an error.
 * (2) Telling a non-capturing stream to wait on an event recorded during
 capture is also an error.
 *
 * So DropoutState's usage syncs could error if an RNN with dropout is called in
 an uncaptured region
 * then called in a captured region (triggering 1), or called in a captured
 region then called # in an uncaptured region (triggering 2).
 *
 * To prevent 1 and 2, lock() only syncs on the last usage event if it was
 recorded in the same
 * capture state as the current state (which also means the same graph, if
 capture is in progress).
 *
 * The solution should be safe as long as capture obeys the following
 restrictions:
 *  - Only one capture may be underway at a time in a given process.
 *  - While a capture is underway, no calls to eager ops on noncapturing streams
 (on any thread)
 *    may interleave with the captured ops.
 *
 * TODO: As people experiment with capture, keep an eye out for use cases that
 might need to
 * relax those restrictions.
 *
 * See https://github.com/pytorch/pytorch/pull/56433 for more discussion.
 */

struct DropoutState {
  // Both buffer and event are lazily instantiated when a dropout state is
  // needed for the first time. Note that in this case needed != used, as we
  // don't need a buffer to e.g. run RNNs in test mode.
  at::Tensor buffer;
  c10::optional<cuda::CUDAEvent> event;
  std::mutex mutex;
#if !defined(USE_ROCM)
  // cudaStreamGetCaptureInfo will never give back a capture id of 0, so 0 can
  // serve as a sentinel value that capture was not underway.
  cuda::CaptureId_t capture_id_last_lock = 0;
  cuda::CaptureId_t capture_id_last_unlock = 0;
#endif

  // Every time we use a dropout state, we need to synchronize with its event,
  // to make sure all previous uses finish running before this one starts. Once
  // we're done, we record the event to allow others to synchronize with this
  // kernel. Those events are really needed only for inter-stream sync on a
  // single GPU. I doubt anyone will want to run cuDNN RNNs in parallel on a
  // single GPU, so they should end up being complete no-ops.
  void lock() {
    // NB: We can't ignore the lock even when event is undefined, because
    // someone could then define it before we get to unlock().
    mutex.lock();
    if (event) {
#if !defined(USE_ROCM)
      // See Note [DropoutState and CUDA graph capture]
      cudaStreamCaptureStatus status;
      AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
          cuda::getCurrentCUDAStream(), &status, &capture_id_last_lock));
      if (status == cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
        capture_id_last_lock = 0;
      }
      if (capture_id_last_lock == capture_id_last_unlock) {
        event->block(cuda::getCurrentCUDAStream());
      }
#else
      event->block(cuda::getCurrentCUDAStream());
#endif
    }
  }

  void unlock() {
    if (event) {
      event->record();
#if !defined(USE_ROCM)
      // See Note [DropoutState and CUDA graph capture]
      cudaStreamCaptureStatus status;
      AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
          cuda::getCurrentCUDAStream(), &status, &capture_id_last_unlock));
      if (status == cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
        capture_id_last_unlock = 0;
      }
      TORCH_INTERNAL_ASSERT(capture_id_last_unlock == capture_id_last_lock);
#endif
    }
    mutex.unlock();
  }
};

DropoutState& get_dropout_state(
    double dropout_p,
    bool train,
    TensorOptions options) {
  // Each state is slightly over 2MB and initialized lazily, so it's fine to
  // cache them.
  static std::vector<DropoutState> dropout_state_cache{
      static_cast<size_t>(cuda::getNumGPUs())};
  static std::mutex state_cache_mut;

  AT_ASSERT(options.device().is_cuda());
  auto device = options.device().index();

  std::unique_lock<std::mutex> lock{state_cache_mut};
  auto& state = dropout_state_cache.at(device);
  if (train && dropout_p > 0) {
    const auto& gen =
        at::detail::getCUDAHooks().getDefaultCUDAGenerator(device);
    auto gen_impl = gen.get<at::CUDAGeneratorImpl>();
    bool reset_rnn_state = gen_impl->reset_rnn_state();
    if (!state.buffer.defined() || reset_rnn_state) {
      std::unique_lock<std::mutex> lock{state.mutex};
      int64_t seed =
          at::empty({}, options.dtype(at::kLong)).random_(gen).item<int64_t>();
      state.buffer = at::_cudnn_init_dropout_state(
          dropout_p, train, seed, options.dtype(at::kByte));
      // NB: CUDA binds the event to a device at creation time, so we can
      // initialize it only now, when we know we're on the correct device.
      if (!state.event.has_value()) {
        state.event.emplace();
      }
    }
  }
  return state;
}

Tensor try_get_weight_buf(
    const Tensor& input,
    TensorList parameters,
    bool has_biases,
    cudnnRNNMode_t mode,
    c10::SymInt hidden_size,
    c10::SymInt proj_size,
    int64_t num_layers,
    bool bidirectional) {
  // Prepare all relevant descriptors
  auto handle = getCudnnHandle();
  auto& any_param = parameters.at(0);
  auto datatype = getCudnnDataType(any_param);

  // Something very naughty is happening here.  try_get_weight_buf
  // is called from _cudnn_impl, which is a *composite*.  In other words,
  // inside the composite function we need to query cudnn to figure out how big
  // the weight buf actually is going to be.  This clearly cannot be done
  // symbolically.  For now, we insert guards here; but once we have the black
  // box handling for dynamic shapes, we could also hypothetically infer out
  // the relationships
  RNNDescriptorParams rnn;
#ifndef USE_CUDNN_RNN_V8_API
  rnn.set(
      mode,
      hidden_size.guard_int(__FILE__, __LINE__),
      proj_size.guard_int(__FILE__, __LINE__),
      num_layers,
      bidirectional,
      promote_rnn_math_type(datatype),
      datatype);
#else
  auto cudnn_input_size = input.size(-1);
  auto packed = false; // eqy: bogus as we do not know if the input is packed
                       // here again, it should also not affect the weights
  rnn.set(
      mode,
      cudnn_input_size,
      packed,
      hidden_size.guard_int(__FILE__, __LINE__),
      proj_size.guard_int(__FILE__, __LINE__),
      num_layers,
      bidirectional,
      promote_rnn_math_type(datatype),
      datatype);
#endif
  RNNDescriptor rnn_desc = rnn.descriptor(handle);

  TensorGeometry x_geom({1, input.sym_size(-1).guard_int(__FILE__, __LINE__)});
  TensorDescriptor x_desc;
  // datatype for x_desc comes from any_param, not input.
  // try_get_weight_buf's job is to check "is the weight buffer correctly laid
  // out for us to run it with input of the same datatype?"
  x_desc.set(datatype, x_geom.sizes(), x_geom.strides(), 5);

#ifndef USE_CUDNN_RNN_V8_API
  auto num_params = get_num_weights(handle, rnn_desc, x_desc, datatype);
#else
  auto num_params = get_num_weights(handle, rnn_desc, datatype);
#endif

  // Try to get parameter storage
  auto param_storage = any_param.storage();
  auto weight_buf = at::empty({0}, any_param.options()).set_(param_storage);
  if (weight_buf.size(0) < num_params) {
    return {};
  } else if (weight_buf.size(0) > num_params) {
    weight_buf = weight_buf.narrow(0, 0, num_params);
  }

  // Get and check data pointers
  auto expected_data_ptrs = get_expected_data_ptrs(
      weight_buf, handle, rnn, rnn_desc, x_desc, datatype);

  int64_t num_parameters = parameters.size();
  int64_t num_ptrs = expected_data_ptrs.size();
  if (proj_size != 0) {
    AT_ASSERT(num_parameters % (has_biases ? 5 : 3) == 0);
    AT_ASSERT(num_ptrs % 5 == 0);
    if (has_biases) {
      AT_ASSERT(num_ptrs == num_parameters);
      for (const auto i : c10::irange(num_parameters)) {
        if (expected_data_ptrs[i] != parameters[i].data_ptr())
          return {};
      }
    } else {
      AT_ASSERT(num_parameters % 3 == 0);
      AT_ASSERT(num_ptrs == num_parameters * 5 / 3);
      for (int64_t param_i = 0, ptr_i = 0; ptr_i < num_ptrs;
           ptr_i += 5, param_i += 3) {
        if (expected_data_ptrs[ptr_i] != parameters[param_i].data_ptr())
          return {};
        if (expected_data_ptrs[ptr_i + 1] != parameters[param_i + 1].data_ptr())
          return {};
        if (expected_data_ptrs[ptr_i + 4] != parameters[param_i + 2].data_ptr())
          return {};
      }
    }
  } else {
    AT_ASSERT(num_ptrs == (num_parameters * (has_biases ? 1 : 2)));
    AT_ASSERT(num_parameters % (has_biases ? 4 : 2) == 0);
    for (int64_t param_i = 0, ptr_i = 0; ptr_i < num_ptrs;
         ptr_i += (has_biases ? 2 : 4), param_i += 2) {
      if (expected_data_ptrs[ptr_i] != parameters[param_i].data_ptr())
        return {};
      if (expected_data_ptrs[ptr_i + 1] != parameters[param_i + 1].data_ptr())
        return {};
    }
  }
  if (!parameters[num_parameters - 1].is_contiguous())
    return {};
  return weight_buf;
}

template <typename hidden_type>
std::pair<Tensor, hidden_type> _cudnn_impl(
    const Tensor& input,
    const Tensor& _batch_sizes,
    const hidden_type& hidden,
    TensorList params,
    bool has_biases,
    cudnnRNNMode_t mode,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  auto [hx, cx] = unpack_hidden(hidden);
  auto hidden_size = hx.sym_size(2);
  SymInt proj_size = 0;
  // For LSTM models with projections hidden size could be different
  if (cx.defined() && cx.sym_size(2) != hx.sym_size(2)) {
    hidden_size = cx.sym_size(2);
    proj_size = hx.sym_size(2);
  }

  // TODO:  try_get_weight_buf returns a Tensor, but _cudnn_rnn below takes a
  // c10::optional<Tensor> in weight_buf's slot.  Do we want try_get_weight_buf
  // to return a c10::optional<Tensor> instead of a defined or undefined Tensor?
  at::cuda::OptionalCUDAGuard guard(input.get_device());
  auto weight_buf = try_get_weight_buf(
      input,
      params,
      has_biases,
      mode,
      hidden_size,
      proj_size,
      num_layers,
      bidirectional);

  TORCH_CHECK(_batch_sizes.dim() == 1, "batch_sizes tensor should be 1D");
  IntArrayRef batch_sizes{
      _batch_sizes.data_ptr<int64_t>(),
      static_cast<size_t>(_batch_sizes.size(0))};

  auto& dropout_state = get_dropout_state(dropout_p, train, input.options());
  std::unique_lock<DropoutState> lock{dropout_state};
  int64_t num_params = has_biases ? 4 : 2;
  if (proj_size != 0) {
    ++num_params;
  }
  auto sym_batch_sizes = c10::SymIntArrayRef(
      reinterpret_cast<const c10::SymInt*>(batch_sizes.data()),
      batch_sizes.size());
  // cudnn_output = std::tuple<output, hy, cy, reserve, new_weight_buf>
  auto cudnn_output = at::_cudnn_rnn_symint(
      input,
      params,
      num_params,
      weight_buf,
      hx,
      cx,
      static_cast<int>(mode),
      hidden_size,
      proj_size,
      num_layers,
      /*batch_first=*/false,
      dropout_p,
      train,
      bidirectional,
      sym_batch_sizes,
      dropout_state.buffer);

  return {
      std::get<0>(cudnn_output),
      pack_hidden<hidden_type>(
          std::get<1>(cudnn_output), std::get<2>(cudnn_output))};
}

template <typename hidden_type>
std::pair<Tensor, hidden_type> _cudnn_impl(
    const Tensor& input,
    const hidden_type& hidden,
    TensorList params,
    bool has_biases,
    cudnnRNNMode_t mode,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  auto [hx, cx] = unpack_hidden(hidden);
  auto hidden_size = hx.sym_size(2);
  c10::SymInt proj_size = 0;
  // For LSTM models with projections hidden size could be different
  if (cx.defined() && cx.sym_size(2) != hx.sym_size(2)) {
    hidden_size = cx.sym_size(2);
    proj_size = hx.sym_size(2);
  }
  at::cuda::OptionalCUDAGuard guard(input.get_device());
  auto weight_buf = try_get_weight_buf(
      input,
      params,
      has_biases,
      mode,
      hidden_size,
      proj_size,
      num_layers,
      bidirectional);
  auto& dropout_state = get_dropout_state(dropout_p, train, input.options());
  std::unique_lock<DropoutState> lock{dropout_state};
  int64_t num_params = has_biases ? 4 : 2;
  if (proj_size != 0) {
    ++num_params;
  }
  // cudnn_output = std::tuple<output, hy, cy, reserve, new_weight_buf>
  auto cudnn_output = at::_cudnn_rnn_symint(
      input,
      params,
      num_params,
      weight_buf,
      hx,
      cx,
      static_cast<int>(mode),
      hidden_size,
      proj_size,
      num_layers,
      batch_first,
      dropout_p,
      train,
      bidirectional,
      /*batch_sizes=*/{},
      dropout_state.buffer);

  return {
      std::get<0>(cudnn_output),
      pack_hidden<hidden_type>(
          std::get<1>(cudnn_output), std::get<2>(cudnn_output))};
}

#define ONE_HIDDEN_RNN(NAME, MODE)                          \
  void NAME##_cudnn(                                        \
      Tensor& output,                                       \
      Tensor& hy,                                           \
      const Tensor& input,                                  \
      const Tensor& hx,                                     \
      TensorList params,                                    \
      bool has_biases,                                      \
      int64_t num_layers,                                   \
      double dropout_p,                                     \
      bool train,                                           \
      bool bidirectional,                                   \
      bool batch_first) {                                   \
    std::tie(output, hy) = _cudnn_impl(                     \
        input,                                              \
        hx,                                                 \
        params,                                             \
        has_biases,                                         \
        MODE,                                               \
        num_layers,                                         \
        dropout_p,                                          \
        train,                                              \
        bidirectional,                                      \
        batch_first);                                       \
  }                                                         \
                                                            \
  void NAME##_packed_cudnn(                                 \
      Tensor& output,                                       \
      Tensor& hy,                                           \
      const Tensor& data,                                   \
      const Tensor& batch_sizes,                            \
      const Tensor& hx,                                     \
      TensorList params,                                    \
      bool has_biases,                                      \
      int64_t num_layers,                                   \
      double dropout_p,                                     \
      bool train,                                           \
      bool bidirectional) {                                 \
    std::tie(output, hy) = _cudnn_impl(                     \
        data,                                               \
        batch_sizes,                                        \
        hx,                                                 \
        params,                                             \
        has_biases,                                         \
        MODE,                                               \
        num_layers,                                         \
        dropout_p,                                          \
        train,                                              \
        bidirectional);                                     \
  }                                                         \
                                                            \
  REGISTER_CUDA_DISPATCH(NAME##_cudnn_stub, &NAME##_cudnn); \
  REGISTER_CUDA_DISPATCH(NAME##_packed_cudnn_stub, &NAME##_packed_cudnn);

ONE_HIDDEN_RNN(gru, CUDNN_GRU)
ONE_HIDDEN_RNN(rnn_tanh, CUDNN_RNN_TANH)
ONE_HIDDEN_RNN(rnn_relu, CUDNN_RNN_RELU)

void lstm_cudnn(
    Tensor& output,
    Tensor& hy,
    Tensor& cy,
    const Tensor& input,
    TensorList hx,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  auto result = _cudnn_impl(
      input,
      std::make_tuple(hx[0], hx[1]),
      params,
      has_biases,
      CUDNN_LSTM,
      num_layers,
      dropout_p,
      train,
      bidirectional,
      batch_first);
  output = result.first;
  hy = std::get<0>(result.second);
  cy = std::get<1>(result.second);
}

void lstm_packed_cudnn(
    Tensor& output,
    Tensor& hy,
    Tensor& cy,
    const Tensor& data,
    const Tensor& batch_sizes,
    TensorList hx,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  auto result = _cudnn_impl(
      data,
      batch_sizes,
      std::make_tuple(hx[0], hx[1]),
      params,
      has_biases,
      CUDNN_LSTM,
      num_layers,
      dropout_p,
      train,
      bidirectional);
  output = result.first;
  hy = std::get<0>(result.second);
  cy = std::get<1>(result.second);
}

REGISTER_CUDA_DISPATCH(lstm_cudnn_stub, &lstm_cudnn);
REGISTER_CUDA_DISPATCH(lstm_packed_cudnn_stub, &lstm_packed_cudnn);

} // namespace

} // namespace at
} // namespace at

#endif // AT_CUDNN_ENABLED()
