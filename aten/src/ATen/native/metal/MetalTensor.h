#include <torch/script.h>

namespace at {
namespace native {
namespace metal {

class MPSImageWrapper;
class MetalTensor final {
  class Impl;

 public:
  MetalTensor(){};
  explicit MetalTensor(const std::vector<int64_t>& sizes);
  explicit MetalTensor(
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides);
  ~MetalTensor() = default;

  MetalTensor(MetalTensor&&) = default;
  MetalTensor& operator=(MetalTensor&&) = default;

  MetalTensor(const MetalTensor&) = default;
  MetalTensor& operator=(const MetalTensor&) = default;

  friend std::ostream& operator<<(std::ostream& output, const MetalTensor& mt);

  static at::Tensor toTensor(MetalTensor&& mt, const TensorOptions& options);
  static MetalTensor& fromTensor(const at::Tensor& tensor);

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

} // namespace metal
} // namespace native
} // namespace at
