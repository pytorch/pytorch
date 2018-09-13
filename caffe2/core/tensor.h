#ifndef CAFFE2_CORE_TENSOR_H_
#define CAFFE2_CORE_TENSOR_H_

#include "caffe2/core/storage.h"
#include "caffe2/core/tensor_impl.h"

#include <ATen/core/intrusive_ptr.h>

namespace caffe2 {

class CAFFE2_API UndefinedTensorImpl final : public TensorImpl {
  UndefinedTensorImpl() : TensorImpl(CPU){};

 public:
 // Without this, we get:
 //  error: identifier "at::UndefinedTensor::_singleton" is undefined in device code
 // (ostensibly because the constexpr tricks MSVC into trying to compile this
 // function for device as well).
#ifdef _WIN32
 static inline TensorImpl * singleton() {
#else
 static constexpr inline TensorImpl * singleton() {
#endif
    return &singleton_;
  }

 private:
  static UndefinedTensorImpl singleton_;
};

/**
 * @brief Tensor class holds a shared pointer to the implementation TensorImpl,
 * redirects API calls to TensorImpl;
 * Copying of Tensor results in sharing the same underlying implementation
 * object
 */
class CAFFE2_API Tensor final {
 protected:
  using TensorImplPtr = c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
  TensorImplPtr impl_;

 public:
  Tensor() : impl_() {}

  operator bool() const {
    return impl_.defined();
  }

  TensorImpl* unsafeGetTensorImpl() const {
    return impl_.get();
  }

  explicit Tensor(DeviceType type)
      : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(type)) {}

  explicit Tensor(const vector<TIndex>& dims, DeviceType type)
      : impl_(
            c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(dims, type)) {}

  explicit Tensor(const vector<int>& dims, DeviceType type)
      : impl_(
            c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(dims, type)) {}

  Tensor(const Tensor& src, BaseContext* context_for_copy, DeviceType type)
      : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
            *src.impl_,
            context_for_copy,
            type)) {}

  Tensor(const Tensor& src, DeviceType type)
      : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
            *src.impl_,
            type)) {}

  template <typename T>
  Tensor(
      const vector<TIndex>& dims,
      const vector<T>& values,
      BaseContext* context)
      : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
            dims,
            values,
            context)) {}

  template <
      typename T,
      typename = typename std::enable_if<std::is_scalar<T>::value>::type>
  Tensor(const T& value, BaseContext* context)
      : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
            value,
            context)) {}

  Tensor Clone() const {
    Tensor x(GetDeviceType());
    x.CopyFrom(*this);
    return x;
  }

  BaseStaticContext* GetStaticContext() const {
    return impl_.get()->GetStaticContext();
  }

  std::unique_ptr<BaseContext> CreateContext() const {
    return impl_.get()->CreateContext();
  }

  DeviceType GetDeviceType() const {
    return impl_.get()->GetDeviceType();
  }

  void CopyFrom(const Tensor& src, BaseContext* context = nullptr) const {
    impl_.get()->CopyFrom(*src.impl_.get(), context);
  }

  void ExtendTo(TIndex num, float growthPct, BaseContext* context) const {
    impl_.get()->ExtendTo(num, growthPct, context);
  }

  void Extend(TIndex num, float growthPct, BaseContext* context) const {
    impl_.get()->Extend(num, growthPct, context);
  }

  void ShrinkTo(TIndex outer_dim) const {
    impl_.get()->ShrinkTo(outer_dim);
  }

  template <class T>
  void ReserveSpace(const T& outer_dim) const {
    impl_.get()->ReserveSpace(outer_dim);
  }

  template <typename... Ts>
  void Resize(Ts... dim_source) const {
    impl_.get()->Resize(dim_source...);
  }

  inline void ResizeLike(const Tensor& src_tensor) const {
    impl_.get()->ResizeLike(*src_tensor.impl_.get());
  }

  inline void Reshape(const vector<TIndex>& dims) const {
    impl_.get()->Reshape(dims);
  }

  inline void Reshape(const vector<int>& dims) const {
    impl_.get()->Reshape(dims);
  }

  inline void FreeMemory() const {
    impl_.get()->FreeMemory();
  }

  string DebugString() const {
    return impl_.get()->DebugString();
  }

  // NB: a.swap(b) is not equivalent to std::swap(a, b);
  // swap method swaps the CONTENTS of the tensors, while std::swap
  // swaps the POINTERS.
  void swap(const Tensor& other) const noexcept {
    impl_.get()->swap(*other.impl_.get());
  }

  void ShareData(const Tensor& src) const {
    impl_.get()->ShareData(*src.impl_.get());
  }

  template <typename T>
  void ShareExternalPointer(
      T* src,
      size_t capacity = 0,
      MemoryDeleter d = nullptr) const {
    impl_.get()->ShareExternalPointer<T>(src, capacity, d);
  }

  template <typename T>
  void ShareExternalPointer(at::DataPtr&& data_ptr, size_t capacity = 0) const {
    impl_.get()->ShareExternalPointer<T>(std::move(data_ptr), capacity);
  }

  void ShareExternalPointer(
      void* src,
      const TypeMeta& meta,
      size_t capacity = 0,
      MemoryDeleter d = nullptr) const {
    impl_.get()->ShareExternalPointer(src, meta, capacity, d);
  }

  void ShareExternalPointer(
      at::DataPtr&& data_ptr,
      const TypeMeta& data_type,
      size_t capacity) {
    impl_.get()->ShareExternalPointer(std::move(data_ptr), data_type, capacity);
  }

  inline const void* raw_data() const {
    return impl_.get()->raw_data();
  }

  template <typename T>
  inline const T* data() const {
    return impl_.get()->data<T>();
  }

  inline void* raw_mutable_data(const TypeMeta& meta) const {
    return impl_.get()->raw_mutable_data(meta);
  }

  inline void* raw_mutable_data() const {
    return impl_.get()->raw_mutable_data();
  }

  template <typename T>
  inline T* mutable_data() const {
    return impl_.get()->mutable_data<T>();
  }

  inline int ndim() const {
    return impl_.get()->ndim();
  }

  inline TIndex size() const {
    return impl_.get()->size();
  }

  inline size_t itemsize() const {
    return impl_.get()->itemsize();
  }

  inline size_t nbytes() const {
    return impl_.get()->nbytes();
  }

  inline size_t capacity_nbytes() const {
    return impl_.get()->capacity_nbytes();
  }

  inline const vector<TIndex>& dims() const {
    return impl_.get()->dims();
  }

  inline TIndex size_from_dim(int k) const {
    return impl_.get()->size_from_dim(k);
  }

  inline TIndex size_to_dim(int k) const {
    return impl_.get()->size_to_dim(k);
  }

  inline TIndex size_between_dim(int k, int l) const {
    return impl_.get()->size_between_dim(k, l);
  }

  inline int canonical_axis_index(int axis_index) const {
    return impl_.get()->canonical_axis_index(axis_index);
  }

  template <typename T>
  inline bool IsType() const {
    return impl_.get()->IsType<T>();
  }

  inline const TypeMeta& meta() const {
    return impl_.get()->meta();
  }

  inline int dim32(const int i) const {
    return impl_.get()->dim32(i);
  }

  inline TIndex dim(const int i) const {
    return impl_.get()->dim(i);
  }

  inline void ExtractDeviceOption(DeviceOption* device) const {
    return impl_.get()->ExtractDeviceOption(device);
  }
};

using TensorCPU = Tensor;

constexpr int k_limit_default_ = 1000;

// TODO: the following logic can be merged into regular Tensor class methods
// after MKLMemory starts to implement Tensor interface

// Type call registry
typedef TypeMeta (*TypeCall)(const void*);
TypeCall GetTypeCallFunction(TypeIdentifier id);
void RegisterTypeCallFunction(TypeIdentifier id, TypeCall c);

// Shape call registry
typedef vector<TIndex> (*TensorInfoCall)(
    const void*,
    size_t* capacity,
    DeviceOption* device);
TensorInfoCall GetTensorInfoFunction(TypeIdentifier id);
void RegisterTensorInfoFunction(TypeIdentifier id, TensorInfoCall c);

// resize helper function
void TensorVectorResize(
    std::vector<Tensor>& tensors,
    int size,
    DeviceType type);

class CAFFE2_API TensorPrinter {
 public:
  explicit TensorPrinter(
      const std::string& tensor_name = "",
      const std::string& file_name = "",
      int limit = k_limit_default_);
  ~TensorPrinter();

  template <class T>
  void Print(const Tensor& tensor);

  void PrintMeta(const Tensor& tensor);

  string MetaStr(const Tensor& tensor);

 private:
  bool to_file_;
  int limit_;
  std::unique_ptr<std::ofstream> log_file_;
  std::string tensor_name_;
};

template <class T>
void TensorPrinter::Print(const Tensor& tensor) {
  std::stringstream values_stream;
  // One most likely doesn't want to print int64-number of items for visual
  // inspection, so we cast down to int here.
  int total_count = static_cast<int>(std::min(tensor.size(), TIndex(limit_)));
  const T* tensor_data = tensor.template data<T>();
  for (int i = 0; i < total_count - 1; ++i) {
    values_stream << tensor_data[i] << ",";
  }
  // We do not add a comma after the last item.
  values_stream << tensor_data[total_count - 1];
  if (to_file_) {
    (*log_file_) << MetaStr(tensor) << values_stream.str() << std::endl;
  } else {
    // Log to console.
    LOG(INFO) << MetaStr(tensor) << values_stream.str();
  }
}

} // namespace caffe2
#endif // CAFFE2_CORE_TENSOR_H_
