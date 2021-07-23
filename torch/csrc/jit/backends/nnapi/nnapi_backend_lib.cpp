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
// TODO T91991928: implement compile() and execute()
class NnapiBackend : public PyTorchBackendInterface {
 public:
  // Constructor.
  explicit NnapiBackend() = default;
  // NOLINTNEXTLINE(modernize-use-override)
  virtual ~NnapiBackend() = default;

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
      weights.get(i).contiguous();
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
    if (comp_ == nullptr) {
      init(handle, inputs);
      std::cout << "init" << std::endl;
    }
    c10::List<at::Tensor> output_list;
    return c10::impl::toList(output_list);
  }

 private:
  std::unique_ptr<torch::nnapi::bind::NnapiCompilation> comp_;
  c10::List<at::Tensor> out_templates_;
  at::Tensor ser_model_;
  mobile::Module shape_compute_module_;

  void init(c10::IValue handle, c10::impl::GenericList inputs) {
    auto dict = handle.toGenericDict();
    std::stringstream ss;
    auto shape_ptr = dict.at("shape_compute_module").toString();
    ss.str(*shape_ptr);
    shape_compute_module_ = _load_for_mobile(ss);

    TORCH_CHECK(comp_ == nullptr);
    auto weights = dict.at("weights").toTensorList();
    out_templates_ = shape_compute_module_.run_method("prepare", ser_model_, weights).toTensorList();
    comp_.reset(new torch::nnapi::bind::NnapiCompilation());
    std::vector<at::Tensor> temp;
    comp_->init(ser_model_, temp);
  }
};

namespace {
constexpr auto backend_name = "nnapi";
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static auto cls = torch::jit::backend<NnapiBackend>(backend_name);
} // namespace

} // namespace jit
} // namespace torch
