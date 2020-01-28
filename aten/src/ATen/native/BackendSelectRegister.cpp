#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/TensorOptions.h>

namespace at {
namespace native {

namespace {

Tensor _cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_cudnn_init_dropout_state", "");
  return op.callUnboxedWithDispatchKey<Tensor, double, bool, int64_t, const TensorOptions &>(key, dropout, train, dropout_seed, options);
}

Tensor arange(Scalar end, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::arange", "");
  return op.callUnboxedWithDispatchKey<Tensor, Scalar, const TensorOptions &>(key, end, options);
}

Tensor arange_start(Scalar start, Scalar end, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::arange", "start");
  return op.callUnboxedWithDispatchKey<Tensor, Scalar, Scalar, const TensorOptions &>(key, start, end, options);
}

Tensor arange_start_step(Scalar start, Scalar end, Scalar step, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::arange", "start_step");
  return op.callUnboxedWithDispatchKey<Tensor, Scalar, Scalar, Scalar, const TensorOptions &>(key, start, end, step, options);
}

Tensor bartlett_window(int64_t window_length, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bartlett_window", "");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, const TensorOptions &>(key, window_length, options);
}

Tensor bartlett_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bartlett_window", "periodic");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, bool, const TensorOptions &>(key, window_length, periodic, options);
}

Tensor blackman_window(int64_t window_length, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::blackman_window", "");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, const TensorOptions &>(key, window_length, options);
}

Tensor blackman_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::blackman_window", "periodic");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, bool, const TensorOptions &>(key, window_length, periodic, options);
}

Tensor empty(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::empty", "memory_format");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, const TensorOptions &, c10::optional<MemoryFormat>>(key, size, options, memory_format);
}

Tensor empty_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  return at::native::empty(size, names, options, memory_format);
}

Tensor empty_strided(IntArrayRef size, IntArrayRef stride, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::empty_strided", "");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, IntArrayRef, const TensorOptions &>(key, size, stride, options);
}

Tensor empty_affine_quantized(IntArrayRef size, const TensorOptions & options, double scale, int64_t zero_point, c10::optional<MemoryFormat> memory_format) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_empty_affine_quantized", "");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, const TensorOptions &, double, int64_t, c10::optional<MemoryFormat>>(key, size, options, scale, zero_point, memory_format);
}

Tensor _empty_per_channel_affine_quantized(IntArrayRef size, const Tensor & scales, const Tensor & zero_points, int64_t axis, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_empty_per_channel_affine_quantized", "");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, const Tensor &, const Tensor &, int64_t, const TensorOptions &, c10::optional<MemoryFormat>>(key, size, scales, zero_points, axis, options, memory_format);
}

Tensor empty_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  return at::native::empty_like(self, options, memory_format);
}

Tensor eye(int64_t n, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::eye", "");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, const TensorOptions &>(key, n, options);
}

Tensor eye_m(int64_t n, int64_t m, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::eye", "m");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, int64_t, const TensorOptions &>(key, n, m, options);
}

Tensor full_names(IntArrayRef size, Scalar fill_value, c10::optional<DimnameList> names, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::full", "names");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, Scalar, c10::optional<DimnameList>, const TensorOptions &>(key, size, fill_value, names, options);
}

Tensor full(IntArrayRef size, Scalar fill_value, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::full", "");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, Scalar, const TensorOptions &>(key, size, fill_value, options);
}

Tensor full_like(const Tensor & self, Scalar fill_value, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::full_like", "dtype");
  return op.callUnboxedWithDispatchKey<Tensor, const Tensor &, Scalar, const TensorOptions &, c10::optional<MemoryFormat>>(key, self, fill_value, options, memory_format);
}

Tensor from_file(std::string filename, c10::optional<bool> shared, c10::optional<int64_t> size, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::from_file", "");
  return op.callUnboxedWithDispatchKey<Tensor, std::string, c10::optional<bool>, c10::optional<int64_t>, const TensorOptions &>(key, filename, shared, size, options);
}

Tensor hann_window(int64_t window_length, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::hann_window", "");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, const TensorOptions &>(key, window_length, options);
}

Tensor hann_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::hann_window", "periodic");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, bool, const TensorOptions &>(key, window_length, periodic, options);
}

Tensor hamming_window(int64_t window_length, const TensorOptions & options={}) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::hann_window", "");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, const TensorOptions &>(key, window_length, options);
}

Tensor hamming_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options={}){
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::hann_window", "periodic");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, bool, const TensorOptions &>(key, window_length, periodic, options);
}

Tensor hamming_window_periodic_alpha(int64_t window_length, bool periodic, double alpha, const TensorOptions & options={}){
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::hann_window", "periodic_alpha");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, bool, double, const TensorOptions &>(key, window_length, periodic, alpha, options);
}

Tensor hamming_window_periodic_alpha_beta(int64_t window_length, bool periodic, double alpha, double beta, const TensorOptions & options={}){
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::hann_window", "periodic_alpha_beta");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, bool, double, double, const TensorOptions &>(key, window_length, periodic, alpha, beta, options);
}

Tensor linspace(Scalar start, Scalar end, int64_t steps, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::linspace", "");
  return op.callUnboxedWithDispatchKey<Tensor, Scalar, Scalar, int64_t, const TensorOptions &>(key, start, end, steps, options);
}

Tensor logspace(Scalar start, Scalar end, int64_t steps, double base, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::logspace", "");
  return op.callUnboxedWithDispatchKey<Tensor, Scalar, Scalar, int64_t, double, const TensorOptions &>(key, start, end, steps, base, options);
}

Tensor ones_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ones", "names");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, c10::optional<DimnameList>, const TensorOptions &>(key, size, names, options);
}

Tensor ones(IntArrayRef size, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ones", "");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, const TensorOptions &>(key, size, options);
}

Tensor ones_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ones_like", "dtype");
  return op.callUnboxedWithDispatchKey<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(key, self, options, memory_format);
}

Tensor scalar_tensor(Scalar s, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::scalar_tensor", "");
  return op.callUnboxedWithDispatchKey<Tensor, Scalar, const TensorOptions &>(key, s, options);
}

Tensor rand_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::rand", "names");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, c10::optional<DimnameList>, const TensorOptions &>(key, size, names, options);
}

Tensor rand_generator_with_names(IntArrayRef size, Generator * generator, c10::optional<DimnameList> names, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::rand", "generator_with_names");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, Generator *, c10::optional<DimnameList>, const TensorOptions &>(key, size, generator, names, options);
}

Tensor rand(IntArrayRef size, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::rand", "");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, const TensorOptions &>(key, size, options);
}

Tensor rand_generator(IntArrayRef size, Generator * generator, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::rand", "generator");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, Generator *, const TensorOptions &>(key, size, generator, options);
}

Tensor rand_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::rand_like", "dtype");
  return op.callUnboxedWithDispatchKey<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(key, self, options, memory_format);
}

Tensor normal(double mean, double std, IntArrayRef size, Generator * generator, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::normal", "float_float");
  return op.callUnboxedWithDispatchKey<Tensor, double, double, IntArrayRef, Generator *, const TensorOptions &>(key, mean, std, size, generator, options);
}

Tensor triu_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::triu_indices", "");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, int64_t, int64_t, const TensorOptions &>(key, row, col, offset, options);
}

Tensor tril_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::tril_indices", "");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, int64_t, int64_t, const TensorOptions &>(key, row, col, offset, options);
}

Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_sparse_coo_tensor_with_dims_and_tensors", "");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, int64_t, IntArrayRef, const Tensor &, const Tensor &, const TensorOptions &>(key, sparse_dim, dense_dim, size, indices, values, options);
}

Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_sparse_coo_tensor_with_dims", "");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, int64_t, IntArrayRef, const TensorOptions &>(key, sparse_dim, dense_dim, size, options);
}

Tensor _sparse_coo_tensor_unsafe(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_sparse_coo_tensor_unsafe", "");
  return op.callUnboxedWithDispatchKey<Tensor, const Tensor &, const Tensor &, IntArrayRef, const TensorOptions &>(key, indices, values, size, options);
}

Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sparse_coo_tensor", "indices_size");
  return op.callUnboxedWithDispatchKey<Tensor, const Tensor &, const Tensor &, IntArrayRef, const TensorOptions &>(key, indices, values, size, options);
}

Tensor sparse_coo_tensor_indices(const Tensor & indices, const Tensor & values, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sparse_coo_tensor", "indices");
  return op.callUnboxedWithDispatchKey<Tensor, const Tensor &, const Tensor &, const TensorOptions &>(key, indices, values, options);
}

Tensor sparse_coo_tensor_size(IntArrayRef size, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sparse_coo_tensor", "size");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, const TensorOptions &>(key, size, options);
}

Tensor zeros_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::zeros_like", "dtype");
  return op.callUnboxedWithDispatchKey<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(key, self, options, memory_format);
}

Tensor zeros(IntArrayRef size, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::zeros", "");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, const TensorOptions &>(key, size, options);
}

Tensor zeros_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::zeros", "names");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, c10::optional<DimnameList>, const TensorOptions &>(key, size, names, options);
}

Tensor range(Scalar start, Scalar end, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::range", "");
  return op.callUnboxedWithDispatchKey<Tensor, Scalar, Scalar, const TensorOptions &>(key, start, end, options);
}

Tensor randperm_generator(int64_t n, Generator * generator, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::randperm", "generator");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, Generator *, const TensorOptions &>(key, n, generator, options);
}

Tensor randperm(int64_t n, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::randperm", "");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, const TensorOptions &>(key, n, options);
}

Tensor randn_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::randn_like", "dtype");
  return op.callUnboxedWithDispatchKey<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(key, self, options, memory_format);
}

Tensor randn_generator_with_names(IntArrayRef size, Generator * generator, c10::optional<DimnameList> names, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::randn", "generator_with_names");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, Generator *, c10::optional<DimnameList>, const TensorOptions &>(key, size, generator, names, options);
}

Tensor randn_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::randn", "names");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, c10::optional<DimnameList>, const TensorOptions &>(key, size, names, options);
}

Tensor randn_generator(IntArrayRef size, Generator * generator, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::randn", "generator");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, Generator *, const TensorOptions &>(key, size, generator, options);
}

Tensor randn(IntArrayRef size, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::randn", "");
  return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, const TensorOptions &>(key, size, options);
}

Tensor randint_like(const Tensor & self, int64_t low, int64_t high, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::randint_like", "low_dtype");
  return op.callUnboxedWithDispatchKey<Tensor, const Tensor &, int64_t, int64_t, const TensorOptions &, c10::optional<MemoryFormat>>(key, self, low, high, options, memory_format);
}

Tensor randint_low(int64_t low, int64_t high, IntArrayRef size, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::randint", "low");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, int64_t, IntArrayRef, const TensorOptions &>(key, low, high, size, options);
}

Tensor randint_generator(int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::randint", "generator");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, IntArrayRef, Generator *, const TensorOptions &>(key, high, size, generator, options);
}

Tensor randint_low_generator(int64_t low, int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::randint", "low_generator");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, int64_t, IntArrayRef, Generator *, const TensorOptions &>(key, low, high, size, generator, options);
}

Tensor randint(int64_t high, IntArrayRef size, const TensorOptions & options) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::randint", "");
  return op.callUnboxedWithDispatchKey<Tensor, int64_t, IntArrayRef, const TensorOptions &>(key, high, size, options);
}

static auto registry = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
    .schema("aten::_cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(_cudnn_init_dropout_state), &_cudnn_init_dropout_state>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(arange_start_step), &arange_start_step>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(arange_start), &arange_start>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(arange), &arange>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(bartlett_window), &bartlett_window>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::bartlett_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(bartlett_window_periodic), &bartlett_window_periodic>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(blackman_window), &blackman_window>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::blackman_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(blackman_window_periodic), &blackman_window_periodic>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(empty), &empty>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::empty_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(empty_like), &empty_like>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::empty.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(empty_names), &empty_names>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_empty_per_channel_affine_quantized(int[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(_empty_per_channel_affine_quantized), &_empty_per_channel_affine_quantized>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(empty_strided), &empty_strided>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(empty_affine_quantized), &empty_affine_quantized>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::eye(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(eye), &eye>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::eye.m(int n, int m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(eye_m), &eye_m>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(scalar_tensor), &scalar_tensor>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_sparse_coo_tensor_unsafe(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(_sparse_coo_tensor_unsafe), &_sparse_coo_tensor_unsafe>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sparse_coo_tensor.indices_size(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(sparse_coo_tensor), &sparse_coo_tensor>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randn_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(randn_like), &randn_like>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randperm.generator(int n, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(randperm_generator), &randperm_generator>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randperm(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(randperm), &randperm>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randint(int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(randint), &randint>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::zeros.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(zeros_names), &zeros_names>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::range(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(range), &range>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sparse_coo_tensor.indices(Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(sparse_coo_tensor_indices), &sparse_coo_tensor_indices>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randint.generator(int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(randint_generator), &randint_generator>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sparse_coo_tensor.size(int[] size, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(sparse_coo_tensor_size), &sparse_coo_tensor_size>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(zeros), &zeros>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::zeros_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(zeros_like), &zeros_like>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rand.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(rand_names), &rand_names>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randint.low_generator(int low, int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(randint_low_generator), &randint_low_generator>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rand.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(rand_generator_with_names), &rand_generator_with_names>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randn.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(randn_generator), &randn_generator>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randint.low(int low, int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(randint_low), &randint_low>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randn(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(randn), &randn>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randint_like.low_dtype(Tensor self, int low, int high, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(randint_like), &randint_like>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(tril_indices), &tril_indices>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_sparse_coo_tensor_with_dims(int sparse_dim, int dense_dim, int[] size, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(_sparse_coo_tensor_with_dims), &_sparse_coo_tensor_with_dims>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, int[] size, Tensor indices, Tensor values, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(_sparse_coo_tensor_with_dims_and_tensors), &_sparse_coo_tensor_with_dims_and_tensors>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(rand), &rand>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rand.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(rand_generator), &rand_generator>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randn.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(randn_generator_with_names), &randn_generator_with_names>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rand_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(rand_like), &rand_like>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ones.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(ones_names), &ones_names>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(ones), &ones>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ones_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(ones_like), &ones_like>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::logspace(Scalar start, Scalar end, int steps=100, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(logspace), &logspace>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randn.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(randn_names), &randn_names>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::linspace(Scalar start, Scalar end, int steps=100, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(linspace), &linspace>(DispatchKey::BackendSelect)
    .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::full_like.dtype(Tensor self, Scalar fill_value, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(full_like), &full_like>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(hann_window), &hann_window>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(from_file), &from_file>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(hann_window_periodic), &hann_window_periodic>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(full), &full>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::full.names(int[] size, Scalar fill_value, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(full_names), &full_names>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(hamming_window), &hamming_window>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::hamming_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(hamming_window_periodic), &hamming_window_periodic>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::hamming_window.periodic_alpha(int window_length, bool periodic, float alpha, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(hamming_window_periodic_alpha), &hamming_window_periodic_alpha>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::hamming_window.periodic_alpha_beta(int window_length, bool periodic, float alpha, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(hamming_window_periodic_alpha_beta), &hamming_window_periodic_alpha_beta>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::normal.float_float(float mean, float std, int[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(normal), &normal>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(triu_indices), &triu_indices>(DispatchKey::BackendSelect)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));

} // namespace
} // native
} // at
