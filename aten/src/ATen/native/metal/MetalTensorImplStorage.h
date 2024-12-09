#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>

namespace at::native::metal {

class MPSImageWrapper;
class MetalTensorImplStorage final {
  class Impl;

 public:
  MetalTensorImplStorage(){};
  MetalTensorImplStorage(const std::vector<int64_t>& sizes);
  MetalTensorImplStorage(
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides);
  ~MetalTensorImplStorage() = default;

  MetalTensorImplStorage(MetalTensorImplStorage&&) = default;
  MetalTensorImplStorage& operator=(MetalTensorImplStorage&&) = default;

  MetalTensorImplStorage(const MetalTensorImplStorage&) = default;
  MetalTensorImplStorage& operator=(const MetalTensorImplStorage&) = default;

  friend std::ostream& operator<<(
      std::ostream& output,
      const MetalTensorImplStorage& mt);

  bool defined() const;
  IntArrayRef sizes() const;
  IntArrayRef strides() const;
  int64_t dim() const;
  int64_t numel() const;
  void set_data_from_host(const float* inputData);
  void copy_data_to_host(float* host);
  MPSImageWrapper* texture() const;

 private:
  std::shared_ptr<Impl> impl();
  std::shared_ptr<const Impl> impl() const;
  std::shared_ptr<Impl> _impl;
};

} // namespace at::native::metal
