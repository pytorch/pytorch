#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "lazy_tensor_core/csrc/cross_replica_reduces.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensor_core/csrc/ir_util.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensor_core/csrc/view.h"
#include "lazy_tensors/computation_client/async_task.h"
#include "lazy_tensors/computation_client/cache.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/multi_wait.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/status.h"
#include "lazy_tensors/types.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_lazy_tensors {

class LazyTensor {
  class DeviceContextArena;
  struct Data;

 public:
  static LazyTensor Create(const at::Tensor& tensor, const Device& device);
  static LazyTensor Create(
      lazy_tensors::ComputationClient::DataPtr handle,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor Create(
      ir::Value ir_value, const Device& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  // Creates an empty/null tensor.
  LazyTensor() = default;

  bool is_null() const { return data_ptr() == nullptr; }

  size_t generation() const { return data()->generation; }

  LazyTensor alias() const { return LazyTensor(data_ptr()); }

  lazy_tensors::int64 size(lazy_tensors::int64 dim) const;

  at::Tensor ToTensor(bool detached);

  void ShallowCopyTo(LazyTensor* dest) const;

  // Assigns the tensor value to the lazy tensor.
  void SetTensor(at::Tensor tensor);

  void UpdateFromTensor(at::Tensor tensor, bool sync);
  void UpdateFromTensorOut(at::Tensor tensor);
  void UpdateFromTensorOut(const LazyTensor& tensor);

  at::ScalarType dtype() const;
  c10::optional<at::ScalarType> dtype_optional() const;

  // Set logical_element_type which is visible to upstream PyTorch.
  void SetScalarType(c10::optional<at::ScalarType> logical_element_type);

  lazy_tensors::util::MaybeRef<lazy_tensors::Shape> shape() const;
  lazy_tensors::Shape shape_with_layout() const;

  const Device& GetDevice() const;
  lazy_tensors::int64 GetUniqueId() const;

  // Retrieves an opaque ID of the alias object upon which the tensor's view is
  // rooted, or 0 if this tensor is not a view.
  std::ptrdiff_t GetViewAliasId() const;

  // Fetches the data behind the tensor. If the tensor has a graph defining
  // its current value, executes the graph and fetches the data result.
  lazy_tensors::ComputationClient::DataPtr GetDataHandle();

  // Fetches the current value of the data, which can be missing (nullptr)
  // in case the tensor has a graph defining its current value,
  lazy_tensors::ComputationClient::DataPtr CurrentDataHandle() const;

  void SetDataHandle(lazy_tensors::ComputationClient::DataPtr handle);

  // Retrieves the current IR Node, or nullptr in case no active IR Node is
  // available.
  ir::Value CurrentIrValue() const;

  // Retrieves the IR Node representing this LazyTensor. One will be created if
  // missing. Note that although this is a const API, it actually changes the
  // internal state ofthe object.
  ir::Value GetIrValue() const;

  c10::optional<at::Tensor> CurrentTensorData() const;

  // Applies the queue of operations in preparation for using the data.
  void ApplyPendingGraph();

  static ir::Value GetDeviceDataIrValue(const at::Scalar& value,
                                        lazy_tensors::PrimitiveType type,
                                        const Device& device);
  static ir::Value GetIrValueForScalar(const at::Scalar& value,
                                       lazy_tensors::PrimitiveType type,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(const at::Scalar& value,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(
      const at::Scalar& value, lazy_tensors::PrimitiveType type,
      lazy_tensors::Span<const lazy_tensors::int64> dimensions,
      const Device& device);
  static ir::Value GetIrValueForScalar(const at::Scalar& value,
                                       const lazy_tensors::Shape& shape,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(
      const at::Scalar& value, const lazy_tensors::Shape& shape,
      c10::optional<at::ScalarType> logical_element_type, const Device& device);

  static ir::Value GetRngSeed(const Device& device);

  static void SetRngSeed(const Device& device, lazy_tensors::uint64 seed);

  static lazy_tensors::uint64 GetRunningSeed(const Device& device);

  // Dispatches a comparison operator, setting the logical type of the result
  // appropriately.
  static LazyTensor DispatchComparisonOp(c10::Symbol kind,
                                         const LazyTensor& input,
                                         const at::Scalar& other);

  // Same as above, with the second input a tensor as well.
  static LazyTensor DispatchComparisonOp(c10::Symbol kind,
                                         const LazyTensor& input,
                                         const LazyTensor& other);

  // Dumps the backend specific text of the computation accumulated in the graph
  // which is attached the tensors.
  static std::string DumpBackendComputation(
      const std::vector<LazyTensor>& tensors);

  // Retrieves the set of lazy tensors which are currently live in the system,
  // for the given device. If device is nullptr, the live tensors for all
  // devices will be returned. Returned tensors are sorted by device as primary
  // key, and by unique ID as secondary key.
  static std::vector<LazyTensor> GetLiveTensors(const Device* device);

  // Applies all the pending IR operations queued over the input tensors. All
  // the tensors must be on the same device. If wait is true, the sync operation
  // will be run synchronously. The devices argument, if not empty, tells the
  // devices which should be partecipating into the replicated computation.
  static void SyncTensorsGraph(std::vector<LazyTensor>* tensors,
                               lazy_tensors::Span<const std::string> devices,
                               bool wait, bool sync_ltc_data);

  // Makes sure that any outstanding IR operation accumulated over live tensors,
  // gets turned into device data. If wait is true, the sync operation will be
  // run synchronously. The devices argument, if not empty, tells the devices
  // which should be partecipating into the replicated computation.
  static void SyncLiveTensorsGraph(
      const Device* device, lazy_tensors::Span<const std::string> devices,
      bool wait);

  // Marks an execution step, which allows the tensor framework to understand
  // the computation boundaries.
  static void MarkStep(const Device& device);

  // Waits for all the outstanding operations on all the supplied devices.
  // If devices is empty, the wait will happen for all local devices.
  static void WaitDeviceOps(lazy_tensors::Span<const std::string> devices);

  // Retrieves the PyTorch CPU tensors behind the lazy tensors IR operations.
  // All the tensors must be on the same device.
  static std::vector<at::Tensor> GetTensors(std::vector<LazyTensor>* tensors);

  // Operation which creates lazy tensors out of PyTorch CPU tensors by batching
  // the requests to the computation servers.
  static std::vector<LazyTensor> CreateTensors(
      const std::vector<at::Tensor>& tensors,
      const std::vector<std::string>& devices);

  //////////////////////////////////////////////////////////////////////////////
  // Special operators follows here, listed in alphabetical order.
  //////////////////////////////////////////////////////////////////////////////
  static std::pair<LazyTensor, ir::Value> all_reduce(
      const LazyTensor& input, const ir::Value& token,
      AllReduceType reduce_type, double scale,
      std::vector<std::vector<lazy_tensors::int64>> groups);

  static ir::Value all_reduce_(
      LazyTensor& input, const ir::Value& token, AllReduceType reduce_type,
      double scale, std::vector<std::vector<lazy_tensors::int64>> groups);

  static ir::Value all_reduce(
      std::vector<LazyTensor>* inputs, const ir::Value& token,
      AllReduceType reduce_type, double scale,
      std::vector<std::vector<lazy_tensors::int64>> groups);

  static std::pair<LazyTensor, ir::Value> all_to_all(
      const LazyTensor& input, const ir::Value& token,
      lazy_tensors::int64 split_dimension, lazy_tensors::int64 concat_dimension,
      lazy_tensors::int64 split_count,
      std::vector<std::vector<lazy_tensors::int64>> groups);

  static std::pair<LazyTensor, ir::Value> collective_permute(
      const LazyTensor& input, const ir::Value& token,
      std::vector<std::pair<lazy_tensors::int64, lazy_tensors::int64>>
          source_target_pairs);

  static LazyTensor get_dimensions_size(
      const LazyTensor& input, std::vector<lazy_tensors::int64> dimensions);

  //////////////////////////////////////////////////////////////////////////////
  // ATEN operators follows here, listed in alphabetical order.
  //////////////////////////////////////////////////////////////////////////////
  static void __ilshift__(LazyTensor& input, const at::Scalar& other);
  static void __ilshift__(LazyTensor& input, const LazyTensor& other);

  static void __irshift__(LazyTensor& input, const at::Scalar& other);
  static void __irshift__(LazyTensor& input, const LazyTensor& other);

  static LazyTensor __lshift__(
      const LazyTensor& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static LazyTensor __lshift__(
      const LazyTensor& input, const LazyTensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor __rshift__(
      const LazyTensor& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static LazyTensor __rshift__(
      const LazyTensor& input, const LazyTensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor adaptive_avg_pool3d(
      const LazyTensor& input, std::vector<lazy_tensors::int64> output_size);

  static LazyTensor adaptive_avg_pool3d_backward(const LazyTensor& grad_output,
                                                 const LazyTensor& input);

  static LazyTensor _adaptive_avg_pool2d(
      const LazyTensor& input, std::vector<lazy_tensors::int64> output_size);

  static LazyTensor _adaptive_avg_pool2d_backward(const LazyTensor& grad_output,
                                                  const LazyTensor& input);

  static void _amp_foreach_non_finite_check_and_unscale_(
      std::vector<LazyTensor> self, LazyTensor& found_inf,
      const LazyTensor& inv_scale);

  static void _amp_update_scale_(LazyTensor& current_scale,
                                 LazyTensor& growth_tracker,
                                 const LazyTensor& found_inf,
                                 double scale_growth_factor,
                                 double scale_backoff_factor,
                                 int growth_interval);

  static LazyTensor abs(const LazyTensor& input);

  static LazyTensor acos(const LazyTensor& input);

  static LazyTensor acosh(const LazyTensor& input);

  static LazyTensor add(
      const LazyTensor& input, const LazyTensor& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static LazyTensor add(
      const LazyTensor& input, const at::Scalar& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static void addcdiv_(LazyTensor& input, const at::Scalar& value,
                       const LazyTensor& tensor1, const LazyTensor& tensor2);

  static LazyTensor addmm(const LazyTensor& input, const LazyTensor& weight,
                          const LazyTensor& bias);

  static LazyTensor all(const LazyTensor& input,
                        std::vector<lazy_tensors::int64> dimensions,
                        bool keep_reduced_dimensions);

  static LazyTensor amax(const LazyTensor& input,
                         std::vector<lazy_tensors::int64> dimensions,
                         bool keep_reduced_dimensions);

  static LazyTensor amin(const LazyTensor& input,
                         std::vector<lazy_tensors::int64> dimensions,
                         bool keep_reduced_dimensions);

  static LazyTensor any(const LazyTensor& input,
                        std::vector<lazy_tensors::int64> dimensions,
                        bool keep_reduced_dimensions);

  static void arange_out(LazyTensor& out, const at::Scalar& start,
                         const at::Scalar& end, const at::Scalar& step,
                         at::ScalarType scalar_type);

  static LazyTensor argmax(const LazyTensor& input, lazy_tensors::int64 dim,
                           bool keepdim);
  static LazyTensor argmax(const LazyTensor& input);

  static LazyTensor argmin(const LazyTensor& input, lazy_tensors::int64 dim,
                           bool keepdim);
  static LazyTensor argmin(const LazyTensor& input);

  // Takes a slice from the input as R1 at the specified offset and reshapes it
  // into the provided size.
  static LazyTensor as_strided(
      const LazyTensor& input, std::vector<lazy_tensors::int64> size,
      std::vector<lazy_tensors::int64> stride,
      c10::optional<lazy_tensors::int64> storage_offset);

  // In-place version of the method above.
  static void as_strided_(LazyTensor& input,
                          std::vector<lazy_tensors::int64> size,
                          std::vector<lazy_tensors::int64> stride,
                          c10::optional<lazy_tensors::int64> storage_offset);

  static LazyTensor asin(const LazyTensor& input);

  static LazyTensor asinh(const LazyTensor& input);

  static LazyTensor atan(const LazyTensor& input);

  static LazyTensor atanh(const LazyTensor& input);

  static LazyTensor atan2(
      const LazyTensor& input, const LazyTensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor avg_pool_nd(const LazyTensor& input,
                                lazy_tensors::int64 spatial_dim_count,
                                std::vector<lazy_tensors::int64> kernel_size,
                                std::vector<lazy_tensors::int64> stride,
                                std::vector<lazy_tensors::int64> padding,
                                bool ceil_mode, bool count_include_pad);

  static LazyTensor avg_pool_nd_backward(
      const LazyTensor& out_backprop, const LazyTensor& input,
      lazy_tensors::int64 spatial_dim_count,
      std::vector<lazy_tensors::int64> kernel_size,
      std::vector<lazy_tensors::int64> stride,
      std::vector<lazy_tensors::int64> padding, bool ceil_mode,
      bool count_include_pad);

  static LazyTensor baddbmm(const LazyTensor& input, const LazyTensor& batch1,
                            const LazyTensor& batch2, const at::Scalar& beta,
                            const at::Scalar& alpha);

  static LazyTensor bernoulli(const LazyTensor& input, double probability);
  static LazyTensor bernoulli(const LazyTensor& input);
  static void bernoulli_(LazyTensor& input, double probability);
  static void bernoulli_(LazyTensor& input, const LazyTensor& probability);

  static LazyTensor binary_cross_entropy(const LazyTensor& input,
                                         const LazyTensor& target,
                                         const LazyTensor& weight,
                                         lazy_tensors::int64 reduction);

  static LazyTensor binary_cross_entropy_backward(
      const LazyTensor& grad_output, const LazyTensor& input,
      const LazyTensor& target, const LazyTensor& weight,
      lazy_tensors::int64 reduction);

  static void logical_and_out(LazyTensor& out, const LazyTensor& input,
                              const LazyTensor& other);

  static LazyTensor bitwise_and(const LazyTensor& input,
                                const at::Scalar& other);

  static LazyTensor bitwise_and(const LazyTensor& input,
                                const LazyTensor& other);

  static void bitwise_not_out(LazyTensor& out, const LazyTensor& input);

  static void bitwise_or_out(LazyTensor& out, const LazyTensor& input,
                             const at::Scalar& other);

  static void bitwise_or_out(LazyTensor& out, const LazyTensor& input,
                             const LazyTensor& other);

  static void bitwise_xor_out(LazyTensor& out, const LazyTensor& input,
                              const at::Scalar& other);

  static void bitwise_xor_out(LazyTensor& out, const LazyTensor& input,
                              const LazyTensor& other);

  // Batch matrix multiplication. Both tensors must be 3D, the batch size must
  // match and the remaining two dimensions must be compatible for matrix
  // multiplication.
  static LazyTensor bmm(const LazyTensor& batch1, const LazyTensor& batch2);

  // Broadcasts the given tensors according to broadcasting semantics.
  static std::vector<LazyTensor> broadcast_tensors(
      lazy_tensors::Span<const LazyTensor> tensors);

  static LazyTensor cat(lazy_tensors::Span<const LazyTensor> tensors,
                        lazy_tensors::int64 dim);

  static LazyTensor ceil(const LazyTensor& input);

  static LazyTensor cholesky(const LazyTensor& input, bool upper);

  static LazyTensor clamp(const LazyTensor& input,
                          const c10::optional<at::Scalar>& min,
                          const c10::optional<at::Scalar>& max);
  static LazyTensor clamp(const LazyTensor& input,
                          const c10::optional<at::Tensor>& min,
                          const c10::optional<at::Tensor>& max);
  static void clamp_out(LazyTensor& out, const LazyTensor& input,
                        const c10::optional<at::Tensor>& min,
                        const c10::optional<at::Tensor>& max);

  static LazyTensor clone(const LazyTensor& input);

  // Pad with the given value and size specified by the given list of low and
  // high paddings.
  static LazyTensor constant_pad_nd(
      const LazyTensor& input,
      lazy_tensors::Span<const lazy_tensors::int64> pad,
      const at::Scalar& value);

  static LazyTensor convolution_overrideable(
      const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
      std::vector<lazy_tensors::int64> stride,
      std::vector<lazy_tensors::int64> padding,
      std::vector<lazy_tensors::int64> dilation, bool transposed,
      std::vector<lazy_tensors::int64> output_padding,
      lazy_tensors::int64 groups);

  static std::tuple<LazyTensor, LazyTensor, LazyTensor>
  convolution_backward_overrideable(
      const LazyTensor& out_backprop, const LazyTensor& input,
      const LazyTensor& weight, std::vector<lazy_tensors::int64> stride,
      std::vector<lazy_tensors::int64> padding,
      std::vector<lazy_tensors::int64> dilation, bool transposed,
      std::vector<lazy_tensors::int64> output_padding,
      lazy_tensors::int64 groups);

  static LazyTensor convolution_overrideable(
      const LazyTensor& input, const LazyTensor& weight,
      std::vector<lazy_tensors::int64> stride,
      std::vector<lazy_tensors::int64> padding,
      std::vector<lazy_tensors::int64> dilation, bool transposed,
      std::vector<lazy_tensors::int64> output_padding,
      lazy_tensors::int64 groups);

  static LazyTensor cosh(const LazyTensor& input);

  // Returns the cross product of the two input tensors in the given dimension.
  // If the dimension is not given, it defaults to the first dimension found
  // with the size 3.
  static LazyTensor cross(const LazyTensor& input, const LazyTensor& other,
                          c10::optional<lazy_tensors::int64> dim);

  // Returns the cumulative product of elements of input in the given dimension.
  static LazyTensor cumprod(const LazyTensor& input, lazy_tensors::int64 dim,
                            c10::optional<at::ScalarType> dtype);

  // Returns the cumulative sum of elements of input in the given dimension.
  static LazyTensor cumsum(const LazyTensor& input, lazy_tensors::int64 dim,
                           c10::optional<at::ScalarType> dtype);

  // If the input is a matrix (2-D tensor), returns a 1-D tensor with the
  // diagonal elements of the input. If the input is a vector (1-D tensor),
  // returns a 2-D square tensor with the elements of input as the diagonal.
  static LazyTensor diag(const LazyTensor& input, lazy_tensors::int64 offset);

  // Returns the diagonal of a matrix (2-D tensor) or batch of matrices. The
  // matrix dimensions are specified by dim1 and dim2, the diagonal by offset.
  static LazyTensor diagonal(const LazyTensor& input,
                             lazy_tensors::int64 offset,
                             lazy_tensors::int64 dim1,
                             lazy_tensors::int64 dim2);

  static LazyTensor div(
      const LazyTensor& input, const LazyTensor& other,
      const c10::optional<c10::string_view>& rounding_mode = c10::nullopt,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static LazyTensor div(const LazyTensor& input, const at::Scalar& other);

  // A generalized contraction between tensors of arbitrary dimension defined by
  // the given equation and applied to the input tensors.
  static LazyTensor einsum(const std::string& equation,
                           lazy_tensors::Span<const LazyTensor> tensors);

  static LazyTensor elu(const LazyTensor& input, const at::Scalar& alpha,
                        const at::Scalar& scale, const at::Scalar& input_scale);
  static void elu_(LazyTensor& input, const at::Scalar& alpha,
                   const at::Scalar& scale, const at::Scalar& input_scale);
  static LazyTensor elu_backward(const LazyTensor& grad_output,
                                 const at::Scalar& alpha,
                                 const at::Scalar& scale,
                                 const at::Scalar& input_scale,
                                 const LazyTensor& output);

  static LazyTensor embedding_dense_backward(const LazyTensor& grad_output,
                                             const LazyTensor& indices,
                                             lazy_tensors::int64 num_weights,
                                             lazy_tensors::int64 padding_idx,
                                             bool scale_grad_by_freq);

  static LazyTensor ts_embedding_dense_backward(const LazyTensor& grad_output,
                                                const LazyTensor& indices,
                                                lazy_tensors::int64 num_weights,
                                                lazy_tensors::int64 padding_idx,
                                                bool scale_grad_by_freq);

  static LazyTensor eq(const LazyTensor& input, const at::Scalar& other);

  static LazyTensor eq(const LazyTensor& input, const LazyTensor& other);

  static LazyTensor erf(const LazyTensor& input);

  static LazyTensor erfc(const LazyTensor& input);

  static LazyTensor erfinv(const LazyTensor& input);

  static LazyTensor exp(const LazyTensor& input);

  static LazyTensor expand(const LazyTensor& input,
                           std::vector<lazy_tensors::int64> size);

  static LazyTensor expm1(const LazyTensor& input);

  static void exponential_(LazyTensor& input, double lambd);

  // Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
  static LazyTensor eye(lazy_tensors::int64 lines, lazy_tensors::int64 cols,
                        const Device& device, at::ScalarType element_type);

  static void eye_out(LazyTensor& out, lazy_tensors::int64 lines,
                      lazy_tensors::int64 cols);

  // Fills the input with the given value.
  static void fill_(LazyTensor& input, const at::Scalar& value);

  // Flips (reverses) the values in the dimensions of the input tensor.
  static LazyTensor flip(const LazyTensor& input,
                         lazy_tensors::Span<const lazy_tensors::int64> dims);

  static LazyTensor floor(const LazyTensor& input);

  static LazyTensor fmod(
      const LazyTensor& input, const LazyTensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static LazyTensor fmod(
      const LazyTensor& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor frac(const LazyTensor& input);

  static LazyTensor full(lazy_tensors::Span<const lazy_tensors::int64> size,
                         const at::Scalar& fill_value, const Device& device,
                         at::ScalarType scalar_type);
  static LazyTensor full_like(const LazyTensor& input,
                              const at::Scalar& fill_value,
                              const Device& device,
                              c10::optional<at::ScalarType> scalar_type);

  static LazyTensor gather(const LazyTensor& input, lazy_tensors::int64 dim,
                           const LazyTensor& index);

  static LazyTensor ge(const LazyTensor& input, const at::Scalar& other);

  static LazyTensor ge(const LazyTensor& input, const LazyTensor& other);

  static LazyTensor gelu(const LazyTensor& input);
  static LazyTensor gelu_backward(const LazyTensor& grad,
                                  const LazyTensor& input);

  static LazyTensor ger(const LazyTensor& input, const LazyTensor& vec2);

  static LazyTensor gt(const LazyTensor& input, const at::Scalar& other);

  static LazyTensor gt(const LazyTensor& input, const LazyTensor& other);

  // Gather slices from input into a result with shape specified by indices. The
  // shape of the indices are first made consistent using broadcast semantics.
  // For input of shape d1 x d2 x ... x dn and p indices of shape i1 x i2 x ...
  // x ik, the output shape is d1 x ... x d(start_dim) x i1 x ... x ik x
  // d(start_dim+p+1) x ... x dn.
  static LazyTensor index(const LazyTensor& input,
                          lazy_tensors::Span<const LazyTensor> indices,
                          lazy_tensors::int64 start_dim);

  static LazyTensor index_add(const LazyTensor& input, lazy_tensors::int64 dim,
                              const LazyTensor& index,
                              const LazyTensor& source);

  static void index_add_(LazyTensor& input, lazy_tensors::int64 dim,
                         const LazyTensor& index, const LazyTensor& source);

  static LazyTensor index_copy(const LazyTensor& input, lazy_tensors::int64 dim,
                               const LazyTensor& index,
                               const LazyTensor& source);

  static void index_copy_(LazyTensor& input, lazy_tensors::int64 dim,
                          const LazyTensor& index, const LazyTensor& source);

  // Fills the elements of the base tensor with the given value in the given
  // dimension, at positions given by the index. The index must be a rank-1
  // tensor.
  static LazyTensor index_fill(const LazyTensor& input, lazy_tensors::int64 dim,
                               const LazyTensor& index,
                               const at::Scalar& value);

  // Same as above, but the value is wrapped as a rank-0 tensor.
  static LazyTensor index_fill(const LazyTensor& input, lazy_tensors::int64 dim,
                               const LazyTensor& index,
                               const LazyTensor& value);

  static void index_fill_(LazyTensor& input, lazy_tensors::int64 dim,
                          const LazyTensor& index, const LazyTensor& value);

  static void index_fill_(LazyTensor& input, lazy_tensors::int64 dim,
                          const LazyTensor& index, const at::Scalar& value);

  // Puts values into the input tensor using the given indices (a tuple of
  // tensors) and returns the result.
  static LazyTensor index_put(
      const LazyTensor& input, lazy_tensors::Span<const LazyTensor> indices,
      lazy_tensors::int64 start_dim, const LazyTensor& values, bool accumulate,
      lazy_tensors::Span<const lazy_tensors::int64> result_permutation);

  static void index_put_(
      LazyTensor& input, const LazyTensor& canonical_base,
      lazy_tensors::Span<const LazyTensor> indices,
      lazy_tensors::int64 start_dim, const LazyTensor& values, bool accumulate,
      lazy_tensors::Span<const lazy_tensors::int64> result_permutation);

  static LazyTensor index_select(const LazyTensor& input,
                                 lazy_tensors::int64 dim,
                                 const LazyTensor& index);

  static LazyTensor inverse(const LazyTensor& input);

  static LazyTensor isnan(const LazyTensor& input);

  static LazyTensor kl_div_backward(const LazyTensor& grad_output,
                                    const LazyTensor& input,
                                    const LazyTensor& target,
                                    lazy_tensors::int64 reduction,
                                    bool log_target);

  static std::tuple<LazyTensor, LazyTensor> kthvalue(const LazyTensor& input,
                                                     lazy_tensors::int64 k,
                                                     lazy_tensors::int64 dim,
                                                     bool keepdim);

  static LazyTensor l1_loss(const LazyTensor& input, const LazyTensor& target,
                            lazy_tensors::int64 reduction);

  static LazyTensor l1_loss_backward(const LazyTensor& grad_output,
                                     const LazyTensor& input,
                                     const LazyTensor& target,
                                     lazy_tensors::int64 reduction);

  static LazyTensor le(const LazyTensor& input, const at::Scalar& other);

  static LazyTensor le(const LazyTensor& input, const LazyTensor& other);

  static LazyTensor hardshrink(const LazyTensor& input,
                               const at::Scalar& lambda);
  static LazyTensor hardshrink_backward(const LazyTensor& grad_out,
                                        const LazyTensor& input,
                                        const at::Scalar& lambda);

  static LazyTensor hardsigmoid(const LazyTensor& input);

  static LazyTensor hardsigmoid_backward(const LazyTensor& grad_output,
                                         const LazyTensor& input);

  static LazyTensor hardtanh_backward(const LazyTensor& grad_output,
                                      const LazyTensor& input,
                                      const at::Scalar& min_val,
                                      const at::Scalar& max_val);

  static LazyTensor leaky_relu(const LazyTensor& input, double negative_slope);
  static LazyTensor leaky_relu_backward(const LazyTensor& grad_output,
                                        const LazyTensor& input,
                                        double negative_slope,
                                        bool self_is_result);
  static LazyTensor lerp(const LazyTensor& input, const LazyTensor& end,
                         const LazyTensor& weight);
  static LazyTensor lerp(const LazyTensor& input, const LazyTensor& end,
                         const at::Scalar& weight);

  static LazyTensor log(const LazyTensor& input);

  static LazyTensor log_base(const LazyTensor& input, ir::OpKind op,
                             double base);

  static LazyTensor log_sigmoid(const LazyTensor& input);
  static std::tuple<LazyTensor, LazyTensor> log_sigmoid_forward(
      const LazyTensor& input);
  static LazyTensor log_sigmoid_backward(const LazyTensor& grad_output,
                                         const LazyTensor& input,
                                         const LazyTensor& buffer);

  static LazyTensor log_softmax(const LazyTensor& input,
                                lazy_tensors::int64 dim,
                                c10::optional<at::ScalarType> dtype);

  static LazyTensor log_softmax_backward(const LazyTensor& grad_output,
                                         const LazyTensor& output,
                                         lazy_tensors::int64 dim);

  static LazyTensor ts_log_softmax_backward(const LazyTensor& grad_output,
                                            const LazyTensor& output,
                                            lazy_tensors::int64 dim,
                                            const LazyTensor& self);

  static LazyTensor log1p(const LazyTensor& input);
  static void log1p_(LazyTensor& input);

  static LazyTensor logdet(const LazyTensor& input);

  static LazyTensor logsumexp(const LazyTensor& input,
                              std::vector<lazy_tensors::int64> dimensions,
                              bool keep_reduced_dimensions);

  static LazyTensor lt(const LazyTensor& input, const at::Scalar& other);

  static LazyTensor lt(const LazyTensor& input, const LazyTensor& other);

  // In-place version of the method above.
  static void masked_fill_(LazyTensor& input, const LazyTensor& mask,
                           const at::Scalar& value);

  static void masked_scatter_(LazyTensor& input, const LazyTensor& mask,
                              const LazyTensor& source);

  static LazyTensor masked_select(const LazyTensor& input,
                                  const LazyTensor& mask);

  static LazyTensor matmul(const LazyTensor& input, const LazyTensor& other);

  static LazyTensor max(
      const LazyTensor& input, const LazyTensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor max(const LazyTensor& input);

  static std::tuple<LazyTensor, LazyTensor> max(const LazyTensor& input,
                                                lazy_tensors::int64 dim,
                                                bool keepdim);

  static void max_out(LazyTensor& max, LazyTensor& max_values,
                      const LazyTensor& input, lazy_tensors::int64 dim,
                      bool keepdim);

  static std::tuple<LazyTensor, LazyTensor> max_pool_nd(
      const LazyTensor& input, lazy_tensors::int64 spatial_dim_count,
      std::vector<lazy_tensors::int64> kernel_size,
      std::vector<lazy_tensors::int64> stride,
      std::vector<lazy_tensors::int64> padding, bool ceil_mode);

  static LazyTensor max_pool_nd_backward(
      const LazyTensor& out_backprop, const LazyTensor& input,
      lazy_tensors::int64 spatial_dim_count,
      std::vector<lazy_tensors::int64> kernel_size,
      std::vector<lazy_tensors::int64> stride,
      std::vector<lazy_tensors::int64> padding, bool ceil_mode);

  static LazyTensor max_unpool(const LazyTensor& input,
                               const LazyTensor& indices,
                               std::vector<lazy_tensors::int64> output_size);

  static LazyTensor max_unpool_backward(
      const LazyTensor& grad_output, const LazyTensor& input,
      const LazyTensor& indices, std::vector<lazy_tensors::int64> output_size);

  static LazyTensor min(
      const LazyTensor& input, const LazyTensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor min(const LazyTensor& input);

  static std::tuple<LazyTensor, LazyTensor> min(const LazyTensor& input,
                                                lazy_tensors::int64 dim,
                                                bool keepdim);

  static void min_out(LazyTensor& min, LazyTensor& min_indices,
                      const LazyTensor& input, lazy_tensors::int64 dim,
                      bool keepdim);

  static LazyTensor mm(const LazyTensor& input, const LazyTensor& weight);

  static LazyTensor mse_loss(const LazyTensor& input, const LazyTensor& target,
                             lazy_tensors::int64 reduction);

  static LazyTensor mse_loss_backward(const LazyTensor& grad_output,
                                      const LazyTensor& input,
                                      const LazyTensor& target,
                                      lazy_tensors::int64 reduction);

  static LazyTensor mul(
      const LazyTensor& input, const LazyTensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static LazyTensor mul(
      const LazyTensor& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor mv(const LazyTensor& input, const LazyTensor& vec);
  static void mv_out(LazyTensor& out, const LazyTensor& input,
                     const LazyTensor& vec);

  // Returns a new tensor that is a narrowed view of the input in the given
  // dimension.
  static LazyTensor narrow(const LazyTensor& input, lazy_tensors::int64 dim,
                           lazy_tensors::int64 start,
                           lazy_tensors::int64 length);

  // Like batch_norm, but returns additional save_mean and save_invstd used by
  // the backward pass.
  static std::tuple<LazyTensor, LazyTensor, LazyTensor> native_batch_norm(
      const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
      LazyTensor& running_mean, LazyTensor& running_var, bool training,
      double momentum, double eps);

  static std::tuple<LazyTensor, LazyTensor, LazyTensor> ts_native_batch_norm(
      const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
      LazyTensor& running_mean, LazyTensor& running_var, bool training,
      double momentum, double eps);

  // Returns the input, weight and bias gradients.
  static std::tuple<LazyTensor, LazyTensor, LazyTensor>
  native_batch_norm_backward(const LazyTensor& grad_out,
                             const LazyTensor& input, const LazyTensor& weight,
                             const LazyTensor& save_mean,
                             const LazyTensor& save_invstd, bool training,
                             double eps);

  static std::tuple<LazyTensor, LazyTensor, LazyTensor>
  ts_native_batch_norm_backward(
      const LazyTensor& grad_out, const LazyTensor& input,
      const LazyTensor& weight, const LazyTensor& running_mean,
      const LazyTensor& running_var, const LazyTensor& save_mean,
      const LazyTensor& save_invstd, bool training, double eps,
      lazy_tensors::Span<const bool> output_mask);

  static LazyTensor ne(const LazyTensor& input, const at::Scalar& other);

  static LazyTensor ne(const LazyTensor& input, const LazyTensor& other);

  static LazyTensor neg(const LazyTensor& input);

  static std::tuple<LazyTensor, LazyTensor>
  nll_loss_forward(const LazyTensor& input, const LazyTensor& target,
      const LazyTensor& weight, lazy_tensors::int64 reduction, int ignore_index);

  static LazyTensor nll_loss2d(const LazyTensor& input,
                               const LazyTensor& target,
                               const LazyTensor& weight,
                               lazy_tensors::int64 reduction, int ignore_index);

  static LazyTensor nll_loss2d_backward(const LazyTensor& grad_output,
                                        const LazyTensor& input,
                                        const LazyTensor& target,
                                        const LazyTensor& weight,
                                        lazy_tensors::int64 reduction,
                                        int ignore_index,
                                        const LazyTensor& total_weight);

  static LazyTensor nll_loss_backward(const LazyTensor& grad_output,
                                      const LazyTensor& input,
                                      const LazyTensor& target,
                                      const LazyTensor& weight,
                                      lazy_tensors::int64 reduction,
                                      int ignore_index,
                                      const LazyTensor& total_weight);

  static std::pair<LazyTensor, LazyTensor> nms(
      const LazyTensor& boxes, const LazyTensor& scores,
      const LazyTensor& score_threshold, const LazyTensor& iou_threshold,
      lazy_tensors::int64 output_size);

  static LazyTensor nonzero(const LazyTensor& input);

  static LazyTensor norm(const LazyTensor& input,
                         const c10::optional<at::Scalar>& p,
                         c10::optional<at::ScalarType> dtype,
                         at::IntArrayRef dim, bool keepdim);

  static LazyTensor normal(double mean, const LazyTensor& std);

  static LazyTensor normal(const LazyTensor& mean, double std);

  static LazyTensor normal(const LazyTensor& mean, const LazyTensor& std);

  static void normal_(LazyTensor& input, double mean, double std);

  static LazyTensor not_supported(std::string description,
                                  lazy_tensors::Shape shape,
                                  const Device& device);

  // Permute the dimensions of this tensor according to the given permutation.
  static LazyTensor permute(const LazyTensor& input,
                            lazy_tensors::Span<const lazy_tensors::int64> dims);

  static LazyTensor pow(const LazyTensor& input, const at::Scalar& exponent);
  static LazyTensor pow(const LazyTensor& input, const LazyTensor& exponent);
  static LazyTensor pow(const at::Scalar& input, const LazyTensor& exponent);

  static LazyTensor prod(const LazyTensor& input,
                         std::vector<lazy_tensors::int64> dimensions,
                         bool keep_reduced_dimensions,
                         c10::optional<at::ScalarType> dtype);

  static void put_(LazyTensor& input, const LazyTensor& index,
                   const LazyTensor& source, bool accumulate);

  static std::tuple<LazyTensor, LazyTensor> qr(const LazyTensor& input,
                                               bool some);

  static void random_(LazyTensor& input);

  static LazyTensor randperm(lazy_tensors::int64 n, const Device& device,
                             at::ScalarType scalar_type);

  static LazyTensor reciprocal(const LazyTensor& input);

  static LazyTensor reflection_pad2d(const LazyTensor& input,
                                     std::vector<lazy_tensors::int64> padding);

  static LazyTensor reflection_pad2d_backward(
      const LazyTensor& grad_output, const LazyTensor& input,
      std::vector<lazy_tensors::int64> padding);

  static LazyTensor relu(const LazyTensor& input);
  static void relu_(LazyTensor& input);

  static LazyTensor remainder(const LazyTensor& input, const LazyTensor& other);
  static LazyTensor remainder(const LazyTensor& input, const at::Scalar& other);

  // Repeats the input tensor along each dimension by the given number of
  // repeats.
  static LazyTensor repeat(const LazyTensor& input,
                           std::vector<lazy_tensors::int64> repeats);

  static LazyTensor replication_pad1d(const LazyTensor& input,
                                      std::vector<lazy_tensors::int64> padding);
  static LazyTensor replication_pad1d_backward(
      const LazyTensor& grad_output, const LazyTensor& input,
      std::vector<lazy_tensors::int64> padding);

  static LazyTensor replication_pad2d(const LazyTensor& input,
                                      std::vector<lazy_tensors::int64> padding);
  static LazyTensor replication_pad2d_backward(
      const LazyTensor& grad_output, const LazyTensor& input,
      std::vector<lazy_tensors::int64> padding);

  static void resize_(LazyTensor& input, std::vector<lazy_tensors::int64> size);

  static LazyTensor round(const LazyTensor& input);

  static LazyTensor rrelu_with_noise(const LazyTensor& input, LazyTensor& noise,
                                     const at::Scalar& lower,
                                     const at::Scalar& upper, bool training);

  static LazyTensor rrelu_with_noise_backward(const LazyTensor& grad_output,
                                              const LazyTensor& input,
                                              const LazyTensor& noise,
                                              const at::Scalar& lower,
                                              const at::Scalar& upper,
                                              bool training);

  static LazyTensor rsqrt(const LazyTensor& input);

  static LazyTensor rsub(
      const LazyTensor& input, const LazyTensor& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static LazyTensor rsub(
      const LazyTensor& input, const at::Scalar& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static void copy_(LazyTensor& input, LazyTensor& src);

  static void scatter_out(LazyTensor& out, const LazyTensor& input,
                          lazy_tensors::int64 dim, const LazyTensor& index,
                          const LazyTensor& src);
  static void scatter_out(LazyTensor& out, const LazyTensor& input,
                          lazy_tensors::int64 dim, const LazyTensor& index,
                          const at::Scalar& value);

  static void scatter_add_(LazyTensor& input, lazy_tensors::int64 dim,
                           const LazyTensor& index, const LazyTensor& src);
  static void scatter_add_out(LazyTensor& out, const LazyTensor& input,
                              lazy_tensors::int64 dim, const LazyTensor& index,
                              const LazyTensor& src);
  static void scatter_add_out(LazyTensor& out, const LazyTensor& input,
                              lazy_tensors::int64 dim, const LazyTensor& index,
                              const at::Scalar& value);

  static LazyTensor select(const LazyTensor& input, lazy_tensors::int64 dim,
                           lazy_tensors::int64 index);

  static void silu_out(LazyTensor& input, LazyTensor& out);
  static LazyTensor sigmoid(const LazyTensor& input);
  static LazyTensor sigmoid_backward(const LazyTensor& grad_output,
                                     const LazyTensor& output);

  static LazyTensor sign(const LazyTensor& input);

  static LazyTensor sin(const LazyTensor& input);

  static LazyTensor sinh(const LazyTensor& input);

  static LazyTensor slice(const LazyTensor& input, lazy_tensors::int64 dim,
                          lazy_tensors::int64 start, lazy_tensors::int64 end,
                          lazy_tensors::int64 step);

  // Computes a loss that uses a squared term if the absolute element-wise error
  // falls below 1 and an L1 term otherwise.
  static LazyTensor smooth_l1_loss(const LazyTensor& input,
                                   const LazyTensor& target,
                                   lazy_tensors::int64 reduction, double beta);

  // Returns the gradient of the input of a smooth_l1_loss operation.
  static LazyTensor smooth_l1_loss_backward(const LazyTensor& grad_output,
                                            const LazyTensor& input,
                                            const LazyTensor& target,
                                            lazy_tensors::int64 reduction,
                                            double beta);

  static LazyTensor softmax(const LazyTensor& input, lazy_tensors::int64 dim,
                            c10::optional<at::ScalarType> dtype);
  static LazyTensor softmax_backward(const LazyTensor& grad_output,
                                     const LazyTensor& output,
                                     lazy_tensors::int64 dim);

  static LazyTensor softplus(const LazyTensor& input, const at::Scalar& beta,
                             const at::Scalar& threshold);
  static LazyTensor softplus_backward(const LazyTensor& grad_output,
                                      const LazyTensor& input,
                                      const at::Scalar& beta,
                                      const at::Scalar& threshold,
                                      const LazyTensor& output);

  static LazyTensor softshrink(const LazyTensor& input,
                               const at::Scalar& lambda);
  static LazyTensor softshrink_backward(const LazyTensor& grad_out,
                                        const LazyTensor& input,
                                        const at::Scalar& lambda);

  static std::vector<LazyTensor> split(const LazyTensor& input,
                                       lazy_tensors::int64 split_size,
                                       lazy_tensors::int64 dim);

  static std::vector<LazyTensor> split_with_sizes(
      const LazyTensor& input, std::vector<lazy_tensors::int64> split_size,
      lazy_tensors::int64 dim);

  static LazyTensor sqrt(const LazyTensor& input);

  // Squeeze out all trivial (size 1) dimensions.
  static LazyTensor squeeze(const LazyTensor& input);

  // Squeeze out the specified dimension index, if trivial (size 1). Returns
  // unchanged input otherwise.
  static LazyTensor squeeze(const LazyTensor& input, lazy_tensors::int64 dim);

  // In-place versions of the methods above.
  static void squeeze_(LazyTensor& input);
  static void squeeze_(LazyTensor& input, lazy_tensors::int64 dim);

  static LazyTensor stack(lazy_tensors::Span<const LazyTensor> tensors,
                          lazy_tensors::int64 dim);

  static LazyTensor std(const LazyTensor& input,
                        std::vector<lazy_tensors::int64> dimensions,
                        bool keep_reduced_dimensions,
                        lazy_tensors::int64 correction);

  static std::tuple<LazyTensor, LazyTensor> std_mean(
      const LazyTensor& input, std::vector<lazy_tensors::int64> dimensions,
      lazy_tensors::int64 correction, bool keep_reduced_dimensions);

  static LazyTensor sub(
      const LazyTensor& input, const LazyTensor& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static LazyTensor sub(
      const LazyTensor& input, const at::Scalar& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static LazyTensor sum(const LazyTensor& input,
                        std::vector<lazy_tensors::int64> dimensions,
                        bool keep_reduced_dimensions,
                        c10::optional<at::ScalarType> dtype);

  static std::tuple<LazyTensor, LazyTensor, LazyTensor> svd(
      const LazyTensor& input, bool some, bool compute_uv);

  static std::tuple<LazyTensor, LazyTensor> symeig(const LazyTensor& input,
                                                   bool eigenvectors,
                                                   bool upper);

  static LazyTensor take(const LazyTensor& input, const LazyTensor& index);

  static LazyTensor tan(const LazyTensor& input);

  static LazyTensor tanh(const LazyTensor& input);
  static LazyTensor tanh_backward(const LazyTensor& grad_output,
                                  const LazyTensor& output);

  static LazyTensor threshold(const LazyTensor& input, float threshold,
                              float value);

  static LazyTensor threshold_backward(const LazyTensor& grad_output,
                                       const LazyTensor& input,
                                       float threshold);

  static LazyTensor to(LazyTensor& input, c10::optional<Device> device,
                       c10::optional<at::ScalarType> scalar_type);

  static std::tuple<LazyTensor, LazyTensor> topk(const LazyTensor& input,
                                                 lazy_tensors::int64 k,
                                                 lazy_tensors::int64 dim,
                                                 bool largest, bool sorted);

  // Returns the sum of the elements of the diagonal of the input 2-D matrix.
  static LazyTensor trace(const LazyTensor& input);

  // Swap given dimensions of the input.
  static LazyTensor transpose(const LazyTensor& input, lazy_tensors::int64 dim0,
                              lazy_tensors::int64 dim1);

  // In-place version of the method above.
  static void transpose_(LazyTensor& input, lazy_tensors::int64 dim0,
                         lazy_tensors::int64 dim1);

  static std::tuple<LazyTensor, LazyTensor> triangular_solve(
      const LazyTensor& rhs, const LazyTensor& lhs, bool left_side, bool upper,
      bool transpose, bool unitriangular);

  // Returns the lower triangular part of a matrix (2-D tensor) or batch of
  // matrices input, the other elements of the result tensor out are set to 0.
  static LazyTensor tril(const LazyTensor& input, lazy_tensors::int64 diagonal);

  // In-place version of the method above.
  static void tril_(LazyTensor& input, lazy_tensors::int64 diagonal);

  // Returns the upper triangular part of a matrix (2-D tensor) or batch of
  // matrices input, the other elements of the result tensor out are set to 0.
  static LazyTensor triu(const LazyTensor& input, lazy_tensors::int64 diagonal);

  // In-place version of the method above.
  static void triu_(LazyTensor& input, lazy_tensors::int64 diagonal);

  static LazyTensor trunc(const LazyTensor& input);

  static LazyTensor ts_softmax_backward(const LazyTensor& grad_output,
                                        const LazyTensor& output,
                                        lazy_tensors::int64 dim,
                                        const LazyTensor& self);

  // Returns a tuple of all slices along a given dimension with that dimension
  // removed.
  static std::vector<LazyTensor> unbind(const LazyTensor& input,
                                        lazy_tensors::int64 dim);

  static void uniform_(LazyTensor& input, double from, double to);

  // Insert a dimension of size one at the specified position.
  static LazyTensor unsqueeze(const LazyTensor& input, lazy_tensors::int64 dim);

  // In-place version of the method above.
  static void unsqueeze_(LazyTensor& input, lazy_tensors::int64 dim);

  static LazyTensor upsample_bilinear2d(
      const LazyTensor& input, std::vector<lazy_tensors::int64> output_size,
      bool align_corners);

  static LazyTensor upsample_bilinear2d_backward(
      const LazyTensor& grad_output,
      std::vector<lazy_tensors::int64> output_size,
      std::vector<lazy_tensors::int64> input_size, bool align_corners);

  static LazyTensor upsample_nearest2d(
      const LazyTensor& input, std::vector<lazy_tensors::int64> output_size);

  static LazyTensor upsample_nearest2d_backward(
      const LazyTensor& grad_output,
      std::vector<lazy_tensors::int64> output_size,
      std::vector<lazy_tensors::int64> input_size);

  static LazyTensor var(const LazyTensor& input,
                        std::vector<lazy_tensors::int64> dimensions,
                        lazy_tensors::int64 correction,
                        bool keep_reduced_dimensions);

  static std::tuple<LazyTensor, LazyTensor> var_mean(
      const LazyTensor& input, std::vector<lazy_tensors::int64> dimensions,
      lazy_tensors::int64 correction, bool keep_reduced_dimensions);

  // Like reshape, but it returns a view into the original tensor.
  static LazyTensor view(
      const LazyTensor& input,
      lazy_tensors::Span<const lazy_tensors::int64> output_size);

  static void zero_(LazyTensor& input);

  static LazyTensor where(const LazyTensor& condition, const LazyTensor& input,
                          const LazyTensor& other);

 private:
  struct SyncTensorsConfig {
    // Whether we want to force data on the target tensors (hence trimming
    // the IR graph above them).
    bool force_ltc_data = true;
    // Whether when setting the data, the other properties of the tensor
    // state should be reset.
    bool sync_ltc_data = true;
  };

  struct SyncTensorCollection {
    SyncTensorCollection() : hash(0) {}

    SyncTensorsConfig config;
    std::vector<size_t> indices;
    lazy_tensors::hash_t hash;
    std::vector<lazy_tensors::util::ExceptionCleanup> unlocker;
    Device device;
  };

  struct PostOrderData {
    std::vector<const ir::Node*> post_order;
    ir::Util::EmissionMap emission_map;
    std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data;
    std::vector<size_t> parameter_sequence;
  };

  struct CompilationResult {
    Device device;
    size_t emitted_nodes = 0;
    std::shared_ptr<lazy_tensors::ComputationClient::Computation> computation;
    std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data;
  };

  struct CachedComputation {
    CachedComputation(
        std::shared_ptr<lazy_tensors::ComputationClient::Computation>
            computation)
        : computation(std::move(computation)) {}

    std::shared_ptr<lazy_tensors::ComputationClient::Computation> computation;
  };

  using ComputationCache =
      lazy_tensors::util::Cache<lazy_tensors::hash_t, CachedComputation,
                                lazy_tensors::util::HashReducer>;

  struct Async {
    Async(SyncTensorCollection* coll,
          std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data,
          std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data,
          ComputationCache::TypePtr cached_computation);

    void Wait();

    lazy_tensors::util::MultiWait mwait;
    std::vector<size_t> indices;
    std::vector<lazy_tensors::util::ExceptionCleanup> unlocker;
    std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data;
    std::string device;
    ComputationCache::TypePtr cached_computation;
    std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data;
  };

  // This is the core lazy tensor data structure where all the tensor data is
  // held. The lazy tensor is nothing more than a shared pointer to a Data
  // object.
  struct Data {
    Data(lazy_tensors::ComputationClient::DataPtr handle, const Device& device,
         c10::optional<at::ScalarType> logical_element_type)
        : handle(std::move(handle)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(ir::Value ir_value, const Device& device,
         c10::optional<at::ScalarType> logical_element_type)
        : ir_value(std::move(ir_value)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(std::shared_ptr<View> view, const Device& device,
         c10::optional<at::ScalarType> logical_element_type)
        : view(std::move(view)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(at::Tensor tensor_data, const Device& device)
        : logical_element_type(tensor_data.scalar_type()),
          tensor_data(std::move(tensor_data)),
          device(device),
          unique_id(GetNextTensorId()) {}

    ~Data();

    lazy_tensors::ComputationClient::DataPtr handle;
    ir::Value ir_value;
    std::shared_ptr<View> view;
    c10::optional<at::ScalarType> logical_element_type;
    c10::optional<at::Tensor> tensor_data;
    const Device device;
    const lazy_tensors::int64 unique_id = 0;
    size_t generation = 1;
  };

  LazyTensor(const at::Tensor& tensor, const Device& device);
  LazyTensor(lazy_tensors::ComputationClient::DataPtr handle,
             c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  LazyTensor(ir::Value ir_value, const Device& device,
             c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  LazyTensor(std::shared_ptr<View> view, const Device& device,
             c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  LazyTensor(std::shared_ptr<Data> data);

  static LazyTensor Create(
      std::shared_ptr<View> view, const Device& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  Data* data() const;

  std::shared_ptr<Data> data_ptr() const { return data_; }

  void SetDataHandle(lazy_tensors::ComputationClient::DataPtr handle,
                     bool sync);

  void SetIrValue(ir::Value ir_value);
  void SetInPlaceIrValue(ir::Value ir_value);

  void AssignIrValue(ir::Value ir_value) const;

  void SetTensorData(at::Tensor tensor_data);

  ir::Value CreateTensorNode(lazy_tensors::ComputationClient::DataPtr data,
                             bool read_only) const;

  View::IrNode GetViewUpdate(const std::shared_ptr<View>& view) const;

  std::shared_ptr<View> UpdateView(std::shared_ptr<View> view,
                                   ir::Value ir_value) const;

  void SetSubView(ViewInfo view_info) const;
  void ModifyCurrentView(ViewInfo view_info) const;
  std::shared_ptr<View> CreateView(ViewInfo view_info) const;
  LazyTensor CreateViewTensor(ViewInfo view_info) const;

  LazyTensor CopyTensorToDevice(const Device& device);

  ir::Value MaybeCastIrValue(
      ir::Value ir_value, const Device& device,
      c10::optional<at::ScalarType> logical_element_type) const;

public:
// TODO(whc) just a hack for now to get codegen to compile... need to refactor
  // Create a new lazy tensor with the same metadata of the input tensor (with
  // possible overrides), and the new IR value.
  LazyTensor CreateFrom(ir::Value ir_value) const;
  LazyTensor CreateFrom(ir::Value ir_value, const Device& device) const;
  LazyTensor CreateFrom(ir::Value ir_value,
                        at::ScalarType logical_element_type) const;
  LazyTensor CreateFrom(
      ir::Value ir_value,
      c10::optional<at::ScalarType> logical_element_type_opt) const;
  LazyTensor CreateFrom(ir::Value ir_value, const Device& device,
                        at::ScalarType logical_element_type) const;

private:
  // We build a graph accumulating operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  std::vector<LazyTensor> MakeOutputTensors(ir::NodePtr node) const;

  ir::Value GetIrValueForTensor(const at::Tensor& tensor,
                                const Device& device) const;

  static ComputationCache* GetComputationCache();

  static SyncTensorCollection CollectSyncTensors(
      const std::vector<LazyTensor>& tensors, const SyncTensorsConfig& config);

  // Implementation of the GetTensors() API using the op-by-op executor.
  static std::vector<at::Tensor> GetTensorsOpByOp(
      std::vector<LazyTensor>* tensors);

  static std::vector<at::Tensor> GetTensorsFused(
      std::vector<LazyTensor>* tensors);

  // Runs an asynchronous syn operation using the op-by-op executor.
  using OpByOpAsync = lazy_tensors::util::AsyncTask<int>;
  static OpByOpAsync SyncTensorsGraphOpByOp(
      std::vector<LazyTensor>* tensors,
      lazy_tensors::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  // Gathers the device data for all the input tensors, after an
  // asynchronous operation.
  static std::vector<lazy_tensors::ComputationClient::DataPtr>
  GatherTensorsData(
      const std::vector<LazyTensor>& tensors,
      lazy_tensors::Span<const size_t> indices,
      lazy_tensors::Span<const lazy_tensors::ComputationClient::DataPtr>
          tensors_data);

  static std::vector<ir::Value> CollectRoots(
      const std::vector<LazyTensor>& tensors,
      lazy_tensors::Span<const size_t> indices);

  static std::vector<lazy_tensors::ComputationClient::DataPtr> FetchTensorData(
      std::vector<LazyTensor>* tensors, const SyncTensorsConfig& config,
      lazy_tensors::Span<const size_t> indices);

  static std::vector<at::Tensor> FetchTensors(
      std::vector<LazyTensor>* tensors,
      lazy_tensors::Span<const lazy_tensors::ComputationClient::DataPtr>
          tensors_data,
      const std::vector<size_t>* indices);

  // Schedules the execution of a sync tensors operation in background. The
  // asynchronous operation will hold the device locks by capturing the ones
  // present within the coll structure.
  static std::shared_ptr<LazyTensor::Async> ScheduleSyncTensorsGraph(
      SyncTensorCollection* coll,
      std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data,
      std::vector<lazy_tensors::ComputationClient::DataPtr> tensors_data,
      ComputationCache::TypePtr cached_computation);

  static std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      std::vector<LazyTensor>* tensors, SyncTensorCollection* coll,
      std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data,
      std::string device, ComputationCache::TypePtr cached_computation);

  static PostOrderData RunPostOrder(const std::vector<LazyTensor>& tensors,
                                    lazy_tensors::Span<const size_t> indices);

  static ComputationCache::TypePtr LookupCachedCompile(
      const std::vector<LazyTensor>& tensors, const lazy_tensors::hash_t& hash);

  static std::shared_ptr<Async> TryRunCachedSync(
      std::vector<LazyTensor>* tensors, SyncTensorCollection* coll,
      PostOrderData* po_data);

  static void BuildInputOutputAliases(const std::vector<LazyTensor>& tensors,
                                      lazy_tensors::Span<const size_t> indices,
                                      ir::LoweringContext* lowering_ctx);

  static CompilationResult Compile(
      const std::vector<LazyTensor>& tensors,
      lazy_tensors::Span<const std::string> devices,
      const SyncTensorCollection& coll, PostOrderData* po_data);

  static std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<LazyTensor>* tensors,
      lazy_tensors::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  static lazy_tensors::int64 GetNextTensorId();

  std::shared_ptr<Data> data_;
};

}  // namespace torch_lazy_tensors
