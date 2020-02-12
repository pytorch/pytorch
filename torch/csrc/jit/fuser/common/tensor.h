#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor_meta.h>

namespace torch {
namespace jit {
namespace fuser {

struct TORCH_API Tensor : public Val {
  ~Tensor() = default;

  Tensor() = delete; //Don't ever want a default constructor, Vals are unique and immutable.
  Tensor(DataType dt)
  : Val(ValType::Tensor, dt), contiguity_(c10::nullopt) {}

  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;

  Tensor(Tensor&& other) = delete;
  Tensor& operator=(Tensor&& other) = delete;


  Tensor(const std::shared_ptr<c10::TensorType>& tensor_type);

  Tensor(const std::shared_ptr<Value>& jit_value);
  
  bool hasContiguityInfo();

  const c10::optional<TensorContiguity>& getContiguityInfo();
protected:

  // Implementation details:
  const c10::optional<TensorContiguity> contiguity_;
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


}}}
