#include <memory>

#include <ATen/nnapi/nnapi_bind.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_exception.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>

namespace torch {
namespace jit {

// This file has no implementation yet, but the declarations are necessary to
// register the backend properly and test preprocess
class NnapiBackend : public PyTorchBackendInterface {
 public:
  // Constructor.
  explicit NnapiBackend() = default;
  ~NnapiBackend() override = default;

  bool is_available() override {
    return true;
  }

  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    auto dict = processed.toGenericDict();

    // make weights contiguous
    auto weights = dict.at("weights").toTensorList();
    for (int i = 0; i < weights.size(); i++) {
      weights[i] = weights.get(i).contiguous();
    }
    dict.insert("weights", weights);

    // save ser_model to member variable
    ser_model_ = dict.at("ser_model").toTensor();

    // wrap procesed in dictionary: {"forward": processed}
    c10::Dict<c10::IValue, c10::IValue> handles(
      c10::StringType::get(), c10::AnyType::get());
    handles.insert("forward", dict);
    return c10::impl::toGenericDict(handles);
  }

  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    c10::List<at::Tensor> tensorInp;
    for (c10::IValue element: inputs) {
      tensorInp.push_back(element.toTensor());
    }
    if (comp_ == nullptr) {
      init(handle, tensorInp);
    }
    TORCH_CHECK(comp_ != nullptr)

    c10::List<at::Tensor> outputs;
    for(at::Tensor out: out_templates_) {
      outputs.push_back(at::empty_like(out));
    }

    // Format inputs
    auto dict = handle.toGenericDict();
    auto inp_mem_fmts = dict.at("inp_mem_fmts").toIntList();
    TORCH_CHECK(tensorInp.size() == inp_mem_fmts.size());
    std::vector<at::Tensor> fixed_inputs;
    for (int i = 0; i < tensorInp.size(); i++) {
      int fmt = inp_mem_fmts[i];
      // These constants match the values in DimOrder in serializer.py
      // TODO: See if it's possible to use those directly.
      if (fmt == 0) {
        fixed_inputs.push_back(tensorInp.get(i).contiguous());
      } else if (fmt == 1) {
        c10::IntArrayRef order = {0, 2, 3, 1};
        fixed_inputs.push_back(tensorInp.get(i).permute(order).contiguous());
      } else {
        throw std::exception();
        std::cerr << "Invalid mem_fmt" << std::endl;
      }
    }

    comp_->run(fixed_inputs, outputs.vec());

    // Format outputs
    auto out_mem_fmts = dict.at("out_mem_fmts").toIntList();
    TORCH_CHECK(outputs.size() == out_mem_fmts.size());
    for (int i = 0; i < outputs.size(); i++) {
      int fmt = out_mem_fmts[i];
      // These constants match the values in DimOrder in serializer.py
      if (fmt == 1) {
        c10::IntArrayRef order = {0, 3, 1, 2};
        outputs[i] = outputs.get(i).permute(order);
      } else if (fmt != 0) {
        throw std::exception();
        std::cerr << "Invalid mem_fmt" << std::endl;
      }
    }

    return c10::impl::toList(outputs);
  }

 private:
  std::unique_ptr<torch::nnapi::bind::NnapiCompilation> comp_;
  c10::List<at::Tensor> out_templates_;
  at::Tensor ser_model_;
  mobile::Module shape_compute_module_;

  void init(c10::IValue handle, c10::List<at::Tensor> inputs) {
    auto dict = handle.toGenericDict();
    std::stringstream ss;
    auto shape_ptr = dict.at("shape_compute_module").toString();
    ss.str(*shape_ptr);
    shape_compute_module_ = _load_for_mobile(ss);

    TORCH_CHECK(comp_ == nullptr);
    auto weights = dict.at("weights").toTensorList();
    out_templates_ = shape_compute_module_.run_method("prepare", ser_model_, inputs).toTensorList();
    comp_.reset(new torch::nnapi::bind::NnapiCompilation());
    comp_->init(ser_model_, weights.vec());
  }
};

namespace {
constexpr auto backend_name = "nnapi";
static auto cls = torch::jit::backend<NnapiBackend>(backend_name);
} // namespace

} // namespace jit
} // namespace torch
