#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor_meta.h>

namespace torch {
namespace jit {
namespace fuser {

struct TORCH_API IterDomain : public Val {
  ~IterDomain() = default;

  IterDomain() = delete;

  IterDomain(
    const Int* _size
  , ParallelType _parallel_method = ParallelType::Serial
  , bool _reduction_domain = false)
  : Val(ValType::IterDomain, DataType::Int)
  , size_(_size)
  , parallel_method_(_parallel_method)
  , reduction_domain_(_reduction_domain) { }
  
  IterDomain(
    const Val* int_size
  , ParallelType _parallel_method = ParallelType::Serial
  , bool _reduction_domain = false)
  : Val(ValType::IterDomain, DataType::Int)
  , size_(static_cast<const Int*> (int_size) )
  , parallel_method_(_parallel_method)
  , reduction_domain_(_reduction_domain) {
    assert(int_size->isVal());
    assert(int_size->getDataType() == DataType::Int);
  }
  

  bool isReduction() const noexcept{return reduction_domain_;}
  ParallelType parallel_method() const noexcept{return parallel_method_;}
  const Int* size() const noexcept {return size_;}

  IterDomain(const IterDomain& other) = delete;
  IterDomain& operator=(const IterDomain& other) = delete;

  IterDomain(IterDomain&& other) = delete;
  IterDomain& operator=(IterDomain&& other) = delete;

private:
  const Int* size_;
  const ParallelType parallel_method_;
  const bool reduction_domain_;
};


struct TORCH_API TensorDomain : public Val {
  ~TensorDomain() = default;

  TensorDomain(const TensorDomain& other) = delete;
  TensorDomain& operator=(const TensorDomain& other) = delete;

  TensorDomain(TensorDomain&& other) = delete;
  TensorDomain& operator=(TensorDomain&& other) = delete;

  TensorDomain(std::vector<const IterDomain*> domain_)
  : Val(ValType::TensorDomain), domain(domain_){}

  const std::vector<const IterDomain*> domain;
};


struct TORCH_API Tensor : public Val {
  ~Tensor() = default;

  Tensor() = delete; //Don't ever want a default constructor, Vals are unique and immutable.

  Tensor(DataType dt, const TensorDomain* _td = nullptr)
  : Val(ValType::Tensor, dt), contiguity_(c10::nullopt), domain(_td) { }

  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;

  Tensor(Tensor&& other) = delete;
  Tensor& operator=(Tensor&& other) = delete;

  Tensor(const std::shared_ptr<c10::TensorType>& tensor_type);

  Tensor(const std::shared_ptr<Value>& jit_value);
  
  bool hasContiguityInfo();

  const c10::optional<TensorContiguity>& getContiguityInfo();

  static const Tensor* MakeDummyTensor(int ndims){
    std::vector<const IterDomain*> sizes;
    for(int i=0; i<ndims; i++){
      sizes.push_back(new IterDomain(new Int()));
    }
    TensorDomain *td = new TensorDomain(sizes);

    return new Tensor(DataType::Float, td);
  }


//protected:

  // Implementation details:
  const c10::optional<TensorContiguity> contiguity_;
  const TensorDomain* domain;
};

struct TORCH_API TensorView : public Val {
  ~TensorView() = default;

  TensorView(const TensorView& other) = delete;
  TensorView& operator=(const TensorView& other) = delete;

  TensorView(TensorView&& other) = delete;
  TensorView& operator=(TensorView&& other) = delete;

  TensorView(const Tensor* _tensor, const TensorDomain* _view)
  : Val(ValType::TensorView), tensor(_tensor), view(_view){}

  const Tensor* tensor;
  const TensorDomain* view;

};

}}}


