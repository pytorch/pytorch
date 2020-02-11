#include <torch/csrc/jit/fuser/common/ir.h>

namespace torch {
namespace jit {
namespace fuser {

struct TORCH_API Tensor : public Val {
  ~Tensor() = default;

  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;

  Tensor(Tensor&& other) = delete;
  Tensor& operator=(Tensor&& other) = delete;

  Tensor()
  : Val(ValType::Tensor){}

};

struct TORCH_API TensorDomain : public Val {
  ~TensorDomain() = default;

  TensorDomain(const TensorDomain& other) = delete;
  TensorDomain& operator=(const TensorDomain& other) = delete;

  TensorDomain(TensorDomain&& other) = delete;
  TensorDomain& operator=(TensorDomain&& other) = delete;

  TensorDomain()
  : Val(ValType::TensorDomain){}
};

struct TORCH_API TensorView : public Val {
  ~TensorView() = default;

  TensorView(const TensorView& other) = delete;
  TensorView& operator=(const TensorView& other) = delete;

  TensorView(TensorView&& other) = delete;
  TensorView& operator=(TensorView&& other) = delete;

  TensorView()
  : Val(ValType::TensorView){}
};

/*
struct TORCH_API Tensor : public Val {
  using VectorInts = std::vector<int64_t>;
  ~Tensor() = default;

  Tensor()
  : Val(ValType::Tensor)
  , scalar_type_(c10::nullopt)
  , sizes_(c10::nullopt)
  , strides_(c10::nullopt) {}

  Tensor(const std::shared_ptr<c10::TensorType>& tensor_type);

  Tensor(const std::shared_ptr<Value>& jit_value);
  
  c10::optional<c10::ScalarType> scalarType() const {
    return scalar_type_;
  };
  c10::optional<VectorInts> sizes() const {
    return sizes_;
  };
  c10::optional<VectorInts> strides() const {
    return strides_;
  };

protected:
  c10::optional<c10::ScalarType> scalar_type_;
  c10::optional<VectorInts> sizes_;
  c10::optional<VectorInts> strides_;
};
*/

}}}