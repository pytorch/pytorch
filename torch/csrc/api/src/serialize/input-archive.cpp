#include <torch/serialize/input-archive.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/script/module.h>
#include <caffe2/serialize/read_adapter_interface.h>
#include <c10/util/Exception.h>

#include <istream>
#include <memory>
#include <string>
#include <utility>

namespace torch {
namespace serialize {

InputArchive::InputArchive() {}

void InputArchive::read(const std::string& key, c10::IValue& ivalue) {
  ivalue = module_.attr(key);
}

bool InputArchive::try_read(
    const std::string& key,
    Tensor& tensor,
    bool is_buffer) {
  if (!module_.hasattr(key)) {
    return false;
  }
  auto iv = module_.attr(key);
  if (!iv.isTensor()) {
    return false;
  }
  auto read_tensor = iv.toTensor();
  // clang-format on
  if (tensor.defined()) {
    torch::NoGradGuard guard;
    if (tensor.device() != read_tensor.device()) {
      tensor.set_data(read_tensor);
    } else {
      tensor.set_(read_tensor);
    }
  } else {
    tensor = std::move(read_tensor);
  }
  return true;
}

void InputArchive::read(
    const std::string& key,
    Tensor& tensor,
    bool is_buffer) {
  TORCH_CHECK(
      try_read(key, tensor, is_buffer),
      "No such serialized tensor '",
      hierarchy_prefix_,
      key,
      "'");
}

bool InputArchive::try_read(const std::string& key, InputArchive& archive) {
  if (!module_.hasattr(key)) {
    return false;
  }
  auto iv = module_.attr(key);
  if (!iv.isModule()) {
    return false;
  }
  archive.module_ = iv.toModule();
  archive.hierarchy_prefix_ = hierarchy_prefix_ + key + ".";
  return true;
}

void InputArchive::read(const std::string& key, InputArchive& archive) {
  TORCH_CHECK(
      try_read(key, archive),
      "No such serialized submodule: '",
      hierarchy_prefix_,
      key,
      "'");
}

void InputArchive::load_from(const std::string& filename,
    c10::optional<torch::Device> device /*= c10::nullopt*/) {
  module_ = torch::jit::load(filename, std::move(device));
}

void InputArchive::load_from(std::istream& stream,
    c10::optional<torch::Device> device /*= c10::nullopt*/) {
  module_ = torch::jit::load(stream, std::move(device));
}

void InputArchive::load_from(
    const char* data,
    size_t size,
    c10::optional<torch::Device> device /*= c10::nullopt*/) {
  using caffe2::serialize::ReadAdapterInterface;
  class OurAdapter : public ReadAdapterInterface {
  public:
    OurAdapter(const char* data, size_t size)
      : data_(data), size_(size) {
    }
    size_t size() const override { return size_; }
    size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override {
      (void) what;
      if (pos >= size_) {
        return 0;
      }
      size_t nread = std::min(static_cast<size_t>(pos) + n, size_) - pos;
      memcpy(buf, data_ + pos, nread);
      return nread;
    }
  private:
    const char* data_;
    size_t size_;
  };
  std::unique_ptr<OurAdapter> adapter(new OurAdapter(data, size));
  module_ = torch::jit::load(std::move(adapter), std::move(device));
}

void InputArchive::load_from(
    const std::function<size_t(uint64_t, void*, size_t)>& read_func,
    const std::function<size_t(void)>& size_func,
    c10::optional<torch::Device> device /*= c10::nullopt*/) {
  using caffe2::serialize::ReadAdapterInterface;
  class OurAdapter : public ReadAdapterInterface {
  public:
    OurAdapter(const std::function<size_t(uint64_t, void*, size_t)>& read_func,
               const std::function<size_t(void)>& size_func)
      : read_func_(read_func),
        size_func_(size_func) {
    }
    size_t size() const override { return size_func_(); }
    size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override {
      (void)what;
      return read_func_(pos, buf, n);
    }
  private:
    const std::function<size_t(uint64_t, void*, size_t)>& read_func_;
    const std::function<size_t(void)>& size_func_;
  };
  std::unique_ptr<OurAdapter> adapter(new OurAdapter(read_func, size_func));
  module_ = torch::jit::load(std::move(adapter), std::move(device));
}

} // namespace serialize
} // namespace torch
