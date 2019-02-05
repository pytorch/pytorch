#include <ATen/native/RNN.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/MatrixRef.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/util/Exception.h>

#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

// See Note [ATen preprocessor philosophy]

Tensor _cudnn_rnn_flatten_weight(
    TensorList weight_arr, int64_t weight_stride0,
    int64_t input_size,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first,
    bool fn_bidirectional
    ) {
  AT_ERROR("_cudnn_rnn_flatten_weight: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _cudnn_rnn(
    const Tensor& input_r,
    TensorList weight, int64_t weight_stride0,
    const Tensor& weight_buf_r, const Tensor& hx, const Tensor& cx,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntArrayRef fn_batch_sizes,
    const Tensor& fn_dropout_state
    ) {
  AT_ERROR("_cudnn_rnn: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _cudnn_rnn_backward(
    const Tensor& input, TensorList weight, int64_t weight_stride0, const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
    const Tensor& output, const Tensor& grad_output_r, const Tensor& grad_hy_r,
    const Tensor& grad_cy_r,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout,
    bool train, bool bidirectional, IntArrayRef batch_sizes,
    const Tensor& dropout_state, const Tensor& reserve,
    std::array<bool, 4> output_mask
    ) {
  AT_ERROR("_cudnn_rnn_backward: ATen not compiled with cuDNN support");
}

Tensor _cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const TensorOptions& options) {
  AT_ERROR("_cudnn_init_dropout_state: ATen not compiled with cuDNN support");
}

}} // namespace at::native

#else // AT_CUDNN_ENABLED()

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

namespace at { namespace native {

namespace {
  // DropoutDescriptor

  struct DropoutDescriptorParams {
    bool train;
    double dropout;
    Tensor dropout_state;
    DropoutDescriptorParams() {}
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
    int64_t hidden_size;
    int64_t num_layers;
    cudnnDirectionMode_t bidirectional;
    cudnnRNNMode_t mode;
    cudnnDataType_t datatype;
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
        default:
        {
          std::ostringstream oss;
          oss << "unrecognized cuDNN RNN mode " << fn_mode;
          AT_ERROR(oss.str());
        }
      }
    }

    void set_bidirectional(bool fn_bidirectional) {
      bidirectional = fn_bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    }

    void set_algo(cudnnRNNAlgo_t algo){
      this->algo = algo;
    }

    void set(int64_t mode, int64_t hidden_size, int64_t num_layers, bool bidirectional, cudnnDataType_t datatype) {
      this->set_mode(mode);
      this->hidden_size = hidden_size;
      this->num_layers = num_layers;
      this->set_bidirectional(bidirectional);
      this->datatype = datatype;
    }


    RNNDescriptor descriptor(cudnnHandle_t handle, DropoutDescriptor&& dropout_desc) const {
      RNNDescriptor rnn_desc;
      rnn_desc.set(handle, hidden_size, num_layers, std::move(dropout_desc), input_mode, bidirectional, mode, datatype, algo);
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
  };

  // TensorDescriptor list

  std::vector<TensorDescriptor> rnn_descriptor_sequence(const Tensor& tensor, IntArrayRef batch_sizes) {
    std::vector<TensorDescriptor> descriptors(batch_sizes.size());
    size_t i = 0;
    // To be mutated in the loop
    auto batch_tensor_size = tensor.sizes().vec();
    for (auto batch_size : batch_sizes) {
      batch_tensor_size[0] = batch_size;
      // NB: cuDNN RNN API does not support 2d descriptors, so we
      // must pad it out to 3d.
      descriptors[i].set(getCudnnDataType(tensor), batch_tensor_size, tensor.strides(), 3);
      i++;
    }
    return descriptors;
  }

  std::vector<TensorDescriptor> rnn_descriptor(const Tensor& tensor, int64_t N) {
    std::vector<TensorDescriptor> descriptors(N);
    for (int64_t i = 0; i < N; i++) {
      descriptors[i].set(tensor, 5);
    }
    return descriptors;
  }

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

    void set(IntArrayRef input_sizes, IntArrayRef batch_sizes_, bool batch_first) {
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

    // TODO: check x for consistency with input_size?
    std::vector<TensorDescriptor> descriptors(Tensor x) const {
      auto is_input_packed = batch_sizes.size() != 0;
      if (is_input_packed) {
        return rnn_descriptor_sequence(x, batch_sizes);
      } else {
        return rnn_descriptor(x[0], seq_length);
      }
    }
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
    std::vector<TensorDescriptor> x_descs;
    std::vector<TensorDescriptor> y_descs;
    TensorDescriptor hx_desc;
    TensorDescriptor hy_desc;
    TensorDescriptor cx_desc;
    TensorDescriptor cy_desc;

    RNNDescriptors(const RNNParams& fn, cudnnHandle_t handle, Tensor x, Tensor y, Tensor hx, Tensor cx) {
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
    std::vector<cudnnTensorDescriptor_t> get_descs(const std::vector<TensorDescriptor>& descs) {
      std::vector<cudnnTensorDescriptor_t> r;
      r.reserve(descs.size());
      for (auto& desc : descs) {
        r.emplace_back(desc.desc());
      }
      return r;
    }

    std::vector<cudnnTensorDescriptor_t> get_x_descs() {
      return get_descs(x_descs);
    }

    std::vector<cudnnTensorDescriptor_t> get_y_descs() {
      return get_descs(y_descs);
    }
  };

  int64_t get_num_weights(cudnnHandle_t handle, const RNNDescriptor& rnn_desc,
                          const TensorDescriptor& x_desc, cudnnDataType_t datatype) {
    size_t weight_size;
    AT_CUDNN_CHECK(cudnnGetRNNParamsSize(handle, rnn_desc.desc(), x_desc.desc(), &weight_size, datatype));
    auto elem_size = dataSize(datatype);
    AT_ASSERTM(weight_size % elem_size == 0, "cudnnGetRNNParamsSize returned nonsensical weight_size");
    return weight_size / elem_size;
  }

  int64_t _num_linear_layers(cudnnRNNMode_t mode) {
    switch(mode) {
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

  /*
    Returns weight and bias tensors for each layer of the RNN. These tensors
    are views on the underlying weight buffer allocated by CuDNN.

    Note: for LSTM and GRU, which have multiple parameters of each type (4 and 3, respectively),
          these parameters are concatenated along the first dimension.
          These parameters are returned in a consistent order by CuDNN:
              (reset, forget, cell, output) for LSTM
              (reset, input, new) for GRU
    Args:
        fn: The RNN function object holding the RNN state
        handle: a CuDNN handle
        weight_buf: a 1D tensor containing the CuDNN-allocated weight (or grad_weight) buffer
    Returns:
        parameters: [(weight_ih, weight_hh, bias_ih, bias_hh)*], with length equal to the num_layers.
            This is represented as a pair of vector, and outer-dimension stride
            (NB: Can't return MatrixRef because we need to allocate the underlying tensor)
  */
  std::pair<std::vector<Tensor>, size_t> // stride0
  get_parameters(
      cudnnHandle_t handle,
      const RNNDescriptorParams& rnn,
      const RNNDescriptor& rnn_desc,
      const TensorDescriptor& x_desc,
      const FilterDescriptor& w_desc,
      const Tensor& weight_buf
  ) {
    auto cudnn_methods = { cudnnGetRNNLinLayerMatrixParams, cudnnGetRNNLinLayerBiasParams };
    std::vector<Tensor> params;
    int64_t num_linear_layers = _num_linear_layers(rnn.mode);
    int64_t num_layers = rnn.num_directions() * rnn.num_layers;
    size_t cur_offset = 0;
    size_t global_layer_params_count = 0;
    for (int64_t layer = 0; layer < num_layers; layer++) {
      size_t layer_params_count = 0;
      for (auto cudnn_method : cudnn_methods) {
        for (int64_t linear_id = 0; linear_id < num_linear_layers; linear_id++) {
          FilterDescriptor lin_layer_mat_desc;
          void* matrix_pointer;
          AT_CUDNN_CHECK(cudnn_method(
                handle,
                rnn_desc.desc(),
                layer,
                x_desc.desc(),
                w_desc.desc(),
                weight_buf.data_ptr(),
                linear_id,
                lin_layer_mat_desc.mut_desc(),
                &matrix_pointer
                ));
          cudnnDataType_t data_type;
          cudnnTensorFormat_t format;
          int nb_dims;
          constexpr int min_dim = 3;
          // TODO: The use of CPU tensor here is a bit goofy in C++,
          // some sort of alloca would be good enough except that it is
          // kind of convenient to be able to prod() on it.
          Tensor filter_dim_a = at::empty(min_dim, at::initialTensorOptions().dtype(kInt));
          AT_CUDNN_CHECK(cudnnGetFilterNdDescriptor(
                lin_layer_mat_desc.desc(),
                min_dim,
                &data_type,
                &format,
                &nb_dims,
                filter_dim_a.data<int>()
                ));

          AT_ASSERTM(nb_dims <= min_dim, "nb_dims = ", nb_dims, "; min_dim  = ", min_dim);
          filter_dim_a = filter_dim_a.slice(0, 0, nb_dims);
          auto elem_size = dataSize(rnn.datatype);
          auto offset_bytes = (char*)matrix_pointer - (char*)weight_buf.data_ptr();
          AT_ASSERTM(offset_bytes % elem_size == 0, "offset_bytes = ", offset_bytes, "; elem_size = ", elem_size);
          size_t offset = offset_bytes / elem_size;

          // for all the RNN types provided by CUDNN, all the ih weights
          // are the same size and are allocated in a contiguous chunk
          // (same for the hh weights, and the ih and hh biases).
          // Since we're storing all the weights in a single tensor anyway,
          // might as well merge the CUDNN ones into a single tensor as well
          int mat_numel = *filter_dim_a.prod(at::ScalarType::Int).data<int>();
          if (linear_id == 0 || linear_id == num_linear_layers / 2) {
            std::initializer_list<int64_t> size = {
              mat_numel * num_linear_layers / 2, 1};
            // Generate a new parameter tensor which is a view into the
            // weight_buf.
            Tensor param = at::empty({0}, weight_buf.options()).set_(weight_buf.storage(), offset, size);
            params.emplace_back(std::move(param));
            layer_params_count++;
          } else {
            AT_ASSERTM(cur_offset == offset, "cur_offset = ", cur_offset, "; offset = ", offset);
          }
          cur_offset = offset + mat_numel;
        }
      } // for cudnn_method
      if (layer == 0) {
        global_layer_params_count = layer_params_count;
      } else {
        AT_ASSERTM(global_layer_params_count == layer_params_count,
                   "global_layer_params_count = ", global_layer_params_count,
                   "; layer_params_count = ", layer_params_count);
      }
    } // for layer
    return std::make_pair(params, global_layer_params_count);
  }

  // This is a lightweight version of the method above used to quickly get the expected
  // parameter offsets.
  std::vector<void*> get_expected_data_ptrs(
        const Tensor& weight_buf, cudnnHandle_t handle, const RNNDescriptorParams& rnn,
        const RNNDescriptor& rnn_desc, const TensorDescriptor& x_desc, cudnnDataType_t datatype) {
    FilterDescriptor w_desc;
    w_desc.set(weight_buf, 3);

    int64_t num_linear_layers = _num_linear_layers(rnn.mode);
    int64_t num_dir_layers = rnn.num_directions() * rnn.num_layers;
    const auto cudnn_methods = { cudnnGetRNNLinLayerMatrixParams, cudnnGetRNNLinLayerBiasParams };
    std::vector<void*> data_ptrs;
    data_ptrs.reserve(num_dir_layers * 2 * 2);
    for (int64_t layer = 0; layer < num_dir_layers; layer++) {
      for (auto cudnn_method : cudnn_methods) {
        // This API returns a separate pointer for weight of every gate,
        // but we represent them as a single tensor, so we're only interested
        // in a very limited subset of possible values.
        const std::array<int64_t, 2> linear_offsets = { 0, num_linear_layers / 2 };
        for (int64_t linear_id : linear_offsets) {
          FilterDescriptor lin_layer_mat_desc;
          void* matrix_pointer;
          AT_CUDNN_CHECK(cudnn_method(
                handle,
                rnn_desc.desc(),
                layer,
                x_desc.desc(),
                w_desc.desc(),
                weight_buf.data_ptr(),
                linear_id,
                lin_layer_mat_desc.mut_desc(),
                &matrix_pointer
                ));
          data_ptrs.push_back(matrix_pointer);
        }
      }
    }
    return data_ptrs;
  }

  void _viewOrCopyParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to, bool copy) {
    AT_ASSERTM(params_from.size(0) == params_to.size(0), "number of layers mismatch");
    for (size_t i = 0; i < params_from.size(0); i++) {
      auto layer_params_from = params_from[i];
      auto layer_params_to = params_to[i];
      // NOTE: these lists have all weights before all biases, so if the layer
      // doesn't use biases, iteration will terminate once layer_params_from ends
      // and ignore them.
      for (auto a = layer_params_from.begin(), b = layer_params_to.begin();
           a != layer_params_from.end() && b != layer_params_to.end();
           ++a, ++b) {
        auto param_from = *a, param_to = *b;
        AT_ASSERTM(param_from.type() == param_to.type(), "parameter types mismatch");
        if (copy) {
            param_to.copy_(param_from.view_as(param_to));
        } else {
            param_from.resize_as_(param_to);
        }
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

  std::vector<int64_t> _hidden_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
    return {rnn.num_layers * rnn.num_directions(), tensors.mini_batch, rnn.hidden_size};
  }

  std::vector<int64_t> _output_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
    if (tensors.is_input_packed()) {
      return {tensors.batch_sizes_sum, rnn.hidden_size * rnn.num_directions()};
    } else {
      return {tensors.seq_length, tensors.mini_batch, rnn.hidden_size * rnn.num_directions()};
    }
  }

  cudnnRNNAlgo_t get_algo(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors){
#if CUDNN_VERSION < 7200 || CUDA_VERSION < 9010
      return CUDNN_RNN_ALGO_STANDARD;
#else
      cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
      const int64_t bsize = tensors.mini_batch;
      //excluding Turing from using persistent rnn.
      if (prop->major == 7 && prop->minor != 5 && rnn.datatype == CUDNN_DATA_HALF && !tensors.is_input_packed()) {
          if (rnn.num_layers == 1 && rnn.hidden_size <= 1024 && rnn.num_directions() == 1 &&
                  rnn.hidden_size % 128 == 0 && tensors.input_size % 128 == 0){
              //technically, batch size should be multiple of 8, but there are quite a few multiple-of-8 batchsizes that give bad perf,
              //weed them out
              if ((bsize % 16 == 0 && bsize != 80 && bsize !=112) || bsize == 8){
                  if ((tensors.seq_length >=40 && bsize <=128) ||
                     (tensors.seq_length >=20 && bsize <=96) ||
                     (tensors.seq_length >=10 && bsize <=32)) {
                     return CUDNN_RNN_ALGO_PERSIST_STATIC;
                  }
              }
          }
      }
      return CUDNN_RNN_ALGO_STANDARD;
#endif
  }

} // anonymous namespace

// NB: does inplace update into TensorList
// It would be a relatively simple matter to refactor this into multiple
// functions, only one of which does an inplace update, but we leave this
// for future work
Tensor _cudnn_rnn_flatten_weight(
    TensorList weight_arr, int64_t weight_stride0,
    int64_t input_size,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first,
    bool fn_bidirectional
    ) {

  AT_CHECK(weight_arr.size() > 0,
           "_cudnn_rnn_flatten_weight_: cannot flatten empty weight list");

  auto any_param = weight_arr[0];

  RNNDescriptorParams rnn;
  rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional, getCudnnDataType(any_param));

  auto handle = getCudnnHandle();
  RNNDescriptor rnn_desc = rnn.descriptor(handle);

  TensorGeometry x_geom({1, input_size});
  TensorDescriptor x_desc;
  x_desc.set(getCudnnDataType(any_param), x_geom.sizes(), x_geom.strides(), 5);

  auto num_weights = get_num_weights(handle, rnn_desc, x_desc, rnn.datatype);
  auto weight_buf = at::zeros(num_weights, any_param.options());

  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);

  // Slice off views into weight_buf
  std::vector<Tensor> params_arr;
  size_t params_stride0;
  std::tie(params_arr, params_stride0) = get_parameters(handle, rnn, rnn_desc, x_desc, w_desc, weight_buf);

  MatrixRef<Tensor> weight{weight_arr, static_cast<size_t>(weight_stride0)},
                    params{params_arr, params_stride0};

  // Copy weights
  _copyParams(weight, params);

  // Update the storage
  for (size_t i = 0; i < weight.size(0); i++) {
    for (auto orig_param_it = weight[i].begin(), new_param_it = params[i].begin();
         orig_param_it != weight[i].end() && new_param_it != params[i].end();
         orig_param_it++, new_param_it++) {
      auto orig_param = *orig_param_it, new_param = *new_param_it;
      orig_param.set_(new_param.view_as(orig_param));
    }
  }

  return weight_buf;
}

// NB: when fn_batch_sizes is empty, that means no batch sizes was specified
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _cudnn_rnn(
    const Tensor& input_r,
    TensorList weight, int64_t weight_stride0,
    const Tensor& weight_buf_r, const Tensor& hx, const Tensor& cx,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntArrayRef fn_batch_sizes,
    const Tensor& fn_dropout_state
    ) {

  check_device(input_r, weight, {hx, cx});
  auto input = input_r;
  auto weight_buf = weight_buf_r;
  if (fn_dropout_state.defined()) {
      auto input_arg = TensorArg(input, "input", 1);
      auto dropout_state_arg = TensorArg(fn_dropout_state, "dropout_states", 15);
      checkSameGPU("cudnn_rnn", input_arg, dropout_state_arg);
  }
  RNNParams fn;
  fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional, getCudnnDataType(input));
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  // TODO: Set device to input

  if (fn.rnn.mode != CUDNN_LSTM) {
    AT_CHECK(!cx.defined(),
             "rnn: illegal defined cx for non-LSTM RNN");
  }

  // TODO: can batch_first be a wrapper around this function?
  auto is_input_packed = fn.tensors.batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
  }

  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  auto output_size = _output_size(fn.rnn, fn.tensors);

  AT_CHECK(hx.is_contiguous(),
           "rnn: hx is not contiguous");
  AT_CHECK(!cx.defined() || cx.is_contiguous(),
           "rnn: cx is not contiguous");

  auto x = input.contiguous();
  auto output = at::empty(output_size, input.options());
  auto hy = at::empty(hidden_size, hx.options());
  Tensor cy;
  if (cx.defined()) {
    cy = at::empty(hidden_size, cx.options());
  } else {
    cy = at::empty({0}, hx.options()); // NB: Not allowed to return undefined tensors
  }
  auto y = output;

  auto handle = getCudnnHandle();
  cudnnRNNAlgo_t algo = get_algo(fn.rnn, fn.tensors);
  fn.rnn.set_algo(algo);
  RNNDescriptors descs(fn, handle, x, y, hx, cx);

  FilterDescriptor w_desc;
  if (!weight_buf.defined()) {
    auto num_weights = get_num_weights(handle, descs.rnn_desc, descs.x_descs[0], fn.rnn.datatype);
    weight_buf = at::empty(num_weights, x.options());
    w_desc.set(weight_buf, 3);
    weight_buf.zero_();
    std::vector<Tensor> params;
    size_t params_stride0;
    std::tie(params, params_stride0) = get_parameters(handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, weight_buf);
    _copyParams(MatrixRef<Tensor>{weight, static_cast<size_t>(weight_stride0)},
                MatrixRef<Tensor>{params, params_stride0});
  } else {
    w_desc.set(weight_buf, 3);
  }

  AT_CHECK(!cx.defined() || cx.sizes().equals(hidden_size),
           "Expected cell size ", IntArrayRef{hidden_size}, ", got ", cx.sizes());

  size_t workspace_size;
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  AT_CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        &workspace_size
        ));
  Tensor workspace = at::empty(workspace_size, input.options().dtype(kByte));

  Tensor reserve;
  // NB: Previously, the test was for fn.requires_grad, but we don't have
  // this information.  Use 'train' as a proxy.
  if (fn_train) {
    size_t reserve_size;
    AT_CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
          handle,
          descs.rnn_desc.desc(),
          fn.tensors.seq_length,
          x_descs_arr.data(),
          &reserve_size
          ));
    reserve = at::empty(reserve_size, input.options().dtype(kByte));
    AT_CUDNN_CHECK(cudnnRNNForwardTraining(
          handle,
          descs.rnn_desc.desc(),
          fn.tensors.seq_length,
          x_descs_arr.data(), x.data_ptr(),
          descs.hx_desc.desc(), hx.data_ptr(),
          descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
          w_desc.desc(), weight_buf.data_ptr(),
          y_descs_arr.data(), y.data_ptr(),
          descs.hy_desc.desc(), hy.data_ptr(),
          descs.cy_desc.desc(), cy.defined() ? cy.data_ptr() : nullptr,
          workspace.data_ptr(), workspace.size(0),
          reserve.data_ptr(), reserve.size(0)
          ));
  } else { // inference
    reserve = at::empty({0}, input.options().dtype(kByte));
    AT_CUDNN_CHECK(cudnnRNNForwardInference(
          handle,
          descs.rnn_desc.desc(),
          fn.tensors.seq_length,
          x_descs_arr.data(), x.data_ptr(),
          descs.hx_desc.desc(), hx.data_ptr(),
          descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
          w_desc.desc(), weight_buf.data_ptr(),
          y_descs_arr.data(), y.data_ptr(),
          descs.hy_desc.desc(), hy.data_ptr(),
          descs.cy_desc.desc(), cy.defined() ? cy.data_ptr() : nullptr,
          workspace.data_ptr(), workspace.size(0)
          ));

  }

  if (batch_first && !is_input_packed) {
    output.transpose_(0, 1);
  }

  return std::make_tuple(output, hy, cy, reserve, weight_buf);
}

std::tuple<Tensor, Tensor, Tensor> _cudnn_rnn_backward_input(
    const Tensor& input_r, const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
    const Tensor& output_r, const Tensor& grad_output_r, const Tensor& grad_hy,
    const Tensor& grad_cy,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntArrayRef fn_batch_sizes,
    const Tensor& fn_dropout_state, const Tensor& fn_reserve,
    std::array<bool, 3> output_mask
    ) {

  auto input = input_r;
  auto grad_output = grad_output_r;
  auto output = output_r;

  RNNParams fn;
  fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional, getCudnnDataType(input));
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  // TODO: Set device to input
  auto handle = getCudnnHandle();

  if (fn.rnn.mode != CUDNN_LSTM) {
    AT_CHECK(!cx.defined(),
             "rnn: illegal defined cx for non-LSTM RNN");
  }

  auto is_input_packed = fn_batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
    grad_output = grad_output.transpose(0, 1);
    output = output.transpose(0, 1);
  }

  auto input_size = _input_size(fn.tensors);
  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  auto output_size = _output_size(fn.rnn, fn.tensors);

  AT_CHECK(hx.is_contiguous(),
           "rnn: hx is not contiguous");
  AT_CHECK(!cx.defined() || cx.is_contiguous(),
           "rnn: cx is not contiguous");

  auto x = input.contiguous();
  auto dy = grad_output.contiguous();
  auto y = output;
  auto w = weight_buf;
  auto dx = at::empty(input.sizes(), input.options()); // TODO: more compact way of saying this
  auto dhy = grad_hy.contiguous().view(hidden_size);
  auto dcy = grad_cy.defined() ? grad_cy.contiguous().view(hidden_size) : Tensor();
  auto dhx = at::empty(hidden_size, hx.options());
  AT_ASSERTM(cx.defined() || !output_mask[2], "illegally required grad of cx for non-LSTM RNN");
  auto dcx = cx.defined() ? at::empty(hidden_size, cx.options()) : Tensor();

  AT_CHECK(fn_train,
           "cudnn RNN backward can only be called in training mode");

  AT_CHECK(input.sizes().equals(input_size),
           "Expected input size ", IntArrayRef{input_size}, ", got ", input.sizes());
  AT_CHECK(output.sizes().equals(output_size),
           "Expected output size ", IntArrayRef{output_size}, ", got ", output.sizes());

  AT_CHECK(!hx.defined() || hx.sizes().equals(hidden_size),
           "Expected hidden size ", IntArrayRef{hidden_size}, ", got ", hx.sizes());
  AT_CHECK(!cx.defined() || cx.sizes().equals(hidden_size),
           "Expected cell size ", IntArrayRef{hidden_size}, ", got ", cx.sizes());
  AT_CHECK(!dhy.defined() || dhy.sizes().equals(hidden_size),
           "Expected d_hidden size ", IntArrayRef{hidden_size}, ", got ", dhy.sizes());
  AT_CHECK(!dcy.defined() || dcy.sizes().equals(hidden_size),
           "Expected d_cell size ", IntArrayRef{hidden_size}, ", got ", dcy.sizes());

  AT_CHECK(dhy.is_cuda() && dy.is_cuda() && (!dcy.defined() || dcy.is_cuda()),
           "Gradients aren't CUDA tensors");

  cudnnRNNAlgo_t algo = get_algo(fn.rnn, fn.tensors);
  fn.rnn.set_algo(algo);
  RNNDescriptors descs(fn, handle, x, y, hx, cx);

  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);

  size_t workspace_size;
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  AT_CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        &workspace_size
        ));
  // TODO: put this in the correct device???
  Tensor workspace = at::empty(workspace_size, input.options().dtype(kByte));

  AT_CUDNN_CHECK(cudnnRNNBackwardData(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        y_descs_arr.data(), y.data_ptr(),
        y_descs_arr.data(), dy.data_ptr(),
        descs.hy_desc.desc(), dhy.data_ptr(),
        descs.cy_desc.desc(), cx.defined() ? dcy.data_ptr() : nullptr,
        w_desc.desc(), w.data_ptr(),
        descs.hx_desc.desc(), hx.data_ptr(),
        descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
        x_descs_arr.data(), dx.data_ptr(),
        descs.hx_desc.desc(), dhx.data_ptr(),
        descs.cx_desc.desc(), cx.defined() ? dcx.data_ptr() : nullptr,
        workspace.data_ptr(), workspace.size(0),
        fn_reserve.data_ptr(), fn_reserve.size(0)
        ));

  if (batch_first && !is_input_packed) {
    dx = dx.transpose_(0, 1);
  }

  return std::make_tuple(dx, dhx, dcx);
}

// NB: This MUST BE CALLED AFTER _cudnn_rnn_backward_input.
// We'll give a user friendly combined function...
std::vector<Tensor> _cudnn_rnn_backward_weight(
    // TODO: I think tensor geometry sufficient for weight_buf/weight
    const Tensor& input_r, TensorList weight_arr, int64_t weight_stride0,
    const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
    const Tensor& output_r,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntArrayRef fn_batch_sizes,
    const Tensor& fn_dropout_state, const Tensor& fn_reserve
    ) {

  MatrixRef<Tensor> weight{ weight_arr, static_cast<size_t>(weight_stride0) };

  auto input = input_r;
  auto output = output_r;

  RNNParams fn;
  fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional, getCudnnDataType(input));
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  auto handle = getCudnnHandle();

  if (fn.rnn.mode != CUDNN_LSTM) {
    AT_CHECK(!cx.defined(),
             "rnn: illegal defined cx for non-LSTM RNN");
  }

  auto is_input_packed = fn_batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
    output = output.transpose(0, 1);
  }

  auto input_size = _input_size(fn.tensors);
  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);

  AT_CHECK(fn_train,
           "cudnn RNN backward can only be called in training mode");

  AT_CHECK(input.sizes().equals(input_size),
           "Expected input size ", IntArrayRef{input_size}, ", got ", input.sizes());
  AT_CHECK(!hx.defined() || hx.sizes().equals(hidden_size),
           "Expected hidden size ", IntArrayRef{hidden_size}, ", got ", hx.sizes());

  // TODO: the above were the only checks in rnn.py, but it doesn't seem
  // like these checks are enough

  AT_CHECK(hx.is_contiguous(),
           "rnn: hx is not contiguous");
  AT_CHECK(!cx.defined() || cx.is_contiguous(),
           "rnn: cx is not contiguous");

  auto x = input.contiguous();
  const auto& y = output;
  auto dw = at::zeros(weight_buf.sizes(), weight_buf.options());

  cudnnRNNAlgo_t algo = get_algo(fn.rnn, fn.tensors);
  fn.rnn.set_algo(algo);
  RNNDescriptors descs(fn, handle, x, y, hx, cx);

  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);

  size_t workspace_size;
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  AT_CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        &workspace_size
        ));
  Tensor workspace = at::empty(workspace_size, input.options().dtype(kByte));

  AT_CUDNN_CHECK(cudnnRNNBackwardWeights(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(), x.data_ptr(),
        descs.hx_desc.desc(), hx.data_ptr(),
        y_descs_arr.data(), y.data_ptr(),
        workspace.data_ptr(), workspace.size(0),
        w_desc.desc(), dw.data_ptr(),
        fn_reserve.data_ptr(), fn_reserve.size(0)
        ));


  std::vector<Tensor> grad_params_arr;
  size_t grad_params_stride0;
  std::tie(grad_params_arr, grad_params_stride0) = get_parameters(handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, dw);
  if (grad_params_stride0 == static_cast<size_t>(weight_stride0)) {
     _viewParams(MatrixRef<Tensor>{grad_params_arr, grad_params_stride0},
              MatrixRef<Tensor>{weight_arr, static_cast<size_t>(weight_stride0)});
      return grad_params_arr;
  } else {
     std::vector<Tensor> grad_weight_arr;
     grad_weight_arr.reserve( weight.numel() );
     for (const auto& w : weight_arr) {
        grad_weight_arr.emplace_back(at::empty(w.sizes(), w.options()));
     }
     _copyParams(MatrixRef<Tensor>{grad_params_arr, grad_params_stride0},
              MatrixRef<Tensor>{grad_weight_arr, static_cast<size_t>(weight_stride0)});
     return grad_weight_arr;
  }
}

// We need this dispatcher because _cudnn_rnn_backward_weight has a stringent
// ordering requirement with _cudnn_rnn_backward_input
std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _cudnn_rnn_backward(
    const Tensor& input, TensorList weight, int64_t weight_stride0, const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
    const Tensor& output, const Tensor& grad_output_r, const Tensor& grad_hy_r,
    const Tensor& grad_cy_r,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout,
    bool train, bool bidirectional, IntArrayRef batch_sizes,
    const Tensor& dropout_state, const Tensor& reserve,
    std::array<bool, 4> output_mask
    ) {

  auto grad_output = grad_output_r.defined() ? grad_output_r : at::zeros_like(output);
  auto grad_hy = grad_hy_r.defined() ? grad_hy_r : at::zeros_like(hx);
  auto grad_cy = cx.defined() ? (grad_cy_r.defined() ? grad_cy_r : at::zeros_like(cx)) : grad_cy_r;

  Tensor dx, dhx, dcx;
  // NB: unconditionally compute this gradient, because it mutates reserve
  std::tie(dx, dhx, dcx) = at::native::_cudnn_rnn_backward_input(input, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, {output_mask[0], output_mask[1], output_mask[2]});
  std::vector<Tensor> dw;
  if (output_mask[3]) {
    dw = at::native::_cudnn_rnn_backward_weight(input, weight, weight_stride0, weight_buf, hx, cx, output, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve);
  }
  return std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>>{dx, dhx, dcx, dw};
}

// TODO: I am not sure if we actually need the 'dropout' and 'train' parameters
// to initialize just the state tensor
Tensor _cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const TensorOptions& options) {
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

std::tuple<Tensor, Tensor> unpack_hidden(const std::tuple<Tensor, Tensor>& hidden) {
  return hidden;
}

template<typename hidden_type>
hidden_type pack_hidden(const Tensor& hx, const Tensor& cx) {
  static_assert(std::is_same<hidden_type, void>::value, "pack_hidden not implemented for this type");
  AT_ERROR("NOT IMPLEMENTED");
}

template<>
Tensor pack_hidden<Tensor>(const Tensor& hx, const Tensor& cx) {
  AT_ASSERT(cx.numel() == 0);
  return hx;
}

template<>
std::tuple<Tensor, Tensor> pack_hidden<std::tuple<Tensor, Tensor>>(const Tensor& hx, const Tensor& cx) {
  return std::make_tuple(hx, cx);
}

struct DropoutState {
  // Both buffer and event are lazily instantiated when a dropout state is needed
  // for the first time. Note that in this case needed != used, as we don't need
  // a bufer to e.g. run RNNs in test mode.
  at::Tensor buffer;
  c10::optional<cuda::CUDAEvent> event;
  std::mutex mutex;

  // Every time we use a dropout state, we need to synchronize with its event,
  // to make sure all previous uses finish running before this one starts. Once
  // we're done, we record the event to allow others to synchronize with this kernel.
  // Those events are really needed only for inter-stream sync on a single GPU.
  // I doubt anyone will want to run cuDNN RNNs in parallel on a single GPU, so
  // they should end up being complete no-ops.
  void lock() {
    // NB: We can't ignore the lock even when event is undefined, because someone
    // could then define it before we get to unlock().
    mutex.lock();
    if (event) {
      event->block(cuda::getCurrentCUDAStream());
    }
  }

  void unlock() {
    if (event) {
      event->record();
    }
    mutex.unlock();
  }
};

DropoutState& get_dropout_state(double dropout_p, bool train, TensorOptions options) {
  // Each state is slightly over 2MB and initialized lazily, so it's fine to cache them.
  static std::vector<DropoutState> ten_dropout_state_cache { static_cast<size_t>(cuda::getNumGPUs()) };
  static std::vector<DropoutState> var_dropout_state_cache { static_cast<size_t>(cuda::getNumGPUs()) };
  static std::mutex state_cache_mut;

  int device = cuda::current_device();
  std::unique_lock<std::mutex> lock {state_cache_mut};
  auto& state = options.is_variable() ? var_dropout_state_cache.at(device)
                                      : ten_dropout_state_cache.at(device);
  if (train && dropout_p > 0 && !state.buffer.defined()) {
    std::unique_lock<std::mutex> lock {state.mutex};
    int64_t seed = at::empty({}, at::kLong).random_().item<int64_t>();
    state.buffer = at::_cudnn_init_dropout_state(
      dropout_p, train, seed, options.dtype(at::kByte));
    // NB: CUDA binds the event to a device at creation time, so we can initialize it
    // only now, when we know we're on the correct device.
    state.event.emplace();
  }
  return state;
}

Tensor try_get_weight_buf(
      const Tensor& input, TensorList parameters, bool has_biases,
      cudnnRNNMode_t mode, int64_t hidden_size, int64_t num_layers, bool bidirectional) {
  // Prepare all relevant descriptors
  auto handle = getCudnnHandle();
  auto datatype = getCudnnDataType(input);

  RNNDescriptorParams rnn;
  rnn.set(mode, hidden_size, num_layers, bidirectional, datatype);
  RNNDescriptor rnn_desc = rnn.descriptor(handle);

  TensorGeometry x_geom ({1, input.size(-1)});
  TensorDescriptor x_desc;
  x_desc.set(datatype, x_geom.sizes(), x_geom.strides(), 5);

  auto num_params = get_num_weights(handle, rnn_desc, x_desc, datatype);

  // Try to get parameter storage
  auto & any_param = parameters.at(0);
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
  AT_ASSERT(num_ptrs == (num_parameters * (has_biases ? 1 : 2)));
  AT_ASSERT(num_ptrs % (has_biases ? 4 : 2) == 0);
  for (int64_t param_i = 0, ptr_i = 0;
       ptr_i < num_ptrs;
       ptr_i += (has_biases ? 2 : 4), param_i += 2) {
    if (expected_data_ptrs[ptr_i] != parameters[param_i].data_ptr()) return {};
    if (expected_data_ptrs[ptr_i + 1] != parameters[param_i + 1].data_ptr()) return {};
  }
  if (!parameters[num_parameters - 1].is_contiguous()) return {};
  return weight_buf;
}

const char * WEIGHT_FORMAT_WARN = "RNN module weights are not part of single contiguous "
                                  "chunk of memory. This means they need to be compacted "
                                  "at every call, possibly greatly increasing memory usage. "
                                  "To compact weights again call flatten_parameters().";

template<typename hidden_type>
std::pair<Tensor, hidden_type> _cudnn_impl(
      const Tensor& input, const Tensor& _batch_sizes, const hidden_type& hidden,
      TensorList params, bool has_biases, cudnnRNNMode_t mode,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  Tensor hx, cx;
  std::tie(hx, cx) = unpack_hidden(hidden);
  int64_t hidden_size = hx.size(2);

  auto weight_buf = try_get_weight_buf(
      input, params, has_biases, mode, hidden_size, num_layers, bidirectional);
  if (!weight_buf.defined()) {
    AT_WARN(WEIGHT_FORMAT_WARN);
  }

  AT_CHECK(_batch_sizes.dim() == 1, "batch_sizes tensor should be 1D");
  IntArrayRef batch_sizes { _batch_sizes.data<int64_t>(), static_cast<size_t>(_batch_sizes.size(0)) };

  auto & dropout_state = get_dropout_state(dropout_p, train, input.options());
  std::unique_lock<DropoutState> lock { dropout_state };
  // cudnn_output = std::tuple<output, hy, cy, reserve, new_weight_buf>
  auto cudnn_output = at::_cudnn_rnn(
      input, params, has_biases ? 4 : 2, weight_buf,
      hx, cx, static_cast<int>(mode), hidden_size, num_layers, /*batch_first=*/false,
      dropout_p, train, bidirectional, batch_sizes, dropout_state.buffer);

  return {std::get<0>(cudnn_output),
          pack_hidden<hidden_type>(std::get<1>(cudnn_output), std::get<2>(cudnn_output))};
}

template<typename hidden_type>
std::pair<Tensor, hidden_type> _cudnn_impl(
      const Tensor& input, const hidden_type& hidden,
      TensorList params, bool has_biases, cudnnRNNMode_t mode,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  Tensor hx, cx;
  std::tie(hx, cx) = unpack_hidden(hidden);
  int64_t hidden_size = hx.size(2);

  auto weight_buf = try_get_weight_buf(
      input, params, has_biases, mode, hidden_size, num_layers, bidirectional);
  if (!weight_buf.defined()) {
    AT_WARN(WEIGHT_FORMAT_WARN);
  }

  auto & dropout_state = get_dropout_state(dropout_p, train, input.options());
  std::unique_lock<DropoutState> lock { dropout_state };
  // cudnn_output = std::tuple<output, hy, cy, reserve, new_weight_buf>
  auto cudnn_output = at::_cudnn_rnn(
      input, params, has_biases ? 4 : 2, weight_buf,
      hx, cx, static_cast<int>(mode), hidden_size, num_layers, batch_first, dropout_p,
      train, bidirectional, /*batch_sizes=*/{}, dropout_state.buffer);

  return {std::get<0>(cudnn_output),
          pack_hidden<hidden_type>(std::get<1>(cudnn_output), std::get<2>(cudnn_output))};
}

#define ONE_HIDDEN_RNN(NAME, MODE)                                             \
void NAME##_cudnn(Tensor& output, Tensor& hy,                                  \
      const Tensor& input, const Tensor& hx,                                   \
      TensorList params, bool has_biases,                                      \
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) { \
  std::tie(output, hy) = _cudnn_impl(input, hx, params, has_biases,            \
      MODE, num_layers, dropout_p, train, bidirectional, batch_first);         \
}                                                                              \
                                                                               \
void NAME##_packed_cudnn(Tensor& output, Tensor& hy,                           \
      const Tensor& data, const Tensor& batch_sizes, const Tensor& hx,         \
      TensorList params, bool has_biases,                                      \
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {  \
  std::tie(output, hy) = _cudnn_impl(data, batch_sizes, hx, params,            \
      has_biases, MODE, num_layers, dropout_p, train, bidirectional);          \
}                                                                              \
                                                                               \
REGISTER_CUDA_DISPATCH(NAME##_cudnn_stub, &NAME##_cudnn);                      \
REGISTER_CUDA_DISPATCH(NAME##_packed_cudnn_stub, &NAME##_packed_cudnn);

ONE_HIDDEN_RNN(gru, CUDNN_GRU)
ONE_HIDDEN_RNN(rnn_tanh, CUDNN_RNN_TANH)
ONE_HIDDEN_RNN(rnn_relu, CUDNN_RNN_RELU)

void lstm_cudnn(Tensor& output, Tensor& hy, Tensor& cy,
      const Tensor& input, TensorList hx,
      TensorList params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  auto result = _cudnn_impl(input, std::make_tuple(hx[0], hx[1]), params, has_biases,
      CUDNN_LSTM, num_layers, dropout_p, train, bidirectional, batch_first);
  output = result.first;
  hy = std::get<0>(result.second);
  cy = std::get<1>(result.second);
}

void lstm_packed_cudnn(Tensor& output, Tensor& hy, Tensor& cy,
      const Tensor& data, const Tensor& batch_sizes, TensorList hx,
      TensorList params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  auto result = _cudnn_impl(data, batch_sizes, std::make_tuple(hx[0], hx[1]),
      params, has_biases, CUDNN_LSTM, num_layers, dropout_p, train, bidirectional);
  output = result.first;
  hy = std::get<0>(result.second);
  cy = std::get<1>(result.second);
}

REGISTER_CUDA_DISPATCH(lstm_cudnn_stub, &lstm_cudnn);
REGISTER_CUDA_DISPATCH(lstm_packed_cudnn_stub, &lstm_packed_cudnn);

} // anonymous namepsace

}} // namespace at::native

#endif // AT_CUDNN_ENABLED()
