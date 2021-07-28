#include <memory>

#include <ATen/nnapi/nnapi_bind.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_exception.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>

namespace torch {
namespace jit {

// Implementation of Android NNAPI Backend delegate

// The Android Neural Networks API (NNAPI) is an Android C API designed
// for running computationally intensive operations for machine learning on
// Android devices. The API is available on all Android devices running
// Android 8.1 (API level 27) or higher.

// Implementation is reflective of caffe2/torch/backends/_nnapi/prepare.py
// NnapiModule.forward()
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

    // Prepare weights
    auto weights = dict.at("weights").toTensorList();
    for (int i = 0; i < weights.size(); i++) {
      weights.set(i, weights.get(i).contiguous());
    }
    dict.insert("weights", weights);

    // Save ser_model to member variable
    ser_model_ = dict.at("ser_model").toTensor();

    // Wrap procesed in dictionary: {"forward": processed}
    c10::Dict<c10::IValue, c10::IValue> handles(
        c10::StringType::get(), c10::AnyType::get());
    handles.insert("forward", dict);
    return c10::impl::toGenericDict(handles);
  }

  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    // Convert inputs to Tensors
    c10::List<at::Tensor> tensorInp;
    for (c10::IValue element : inputs) {
      tensorInp.push_back(element.toTensor());
    }

    // Lazily call init()
    if (comp_ == nullptr) {
      init(handle, tensorInp);
    }
    TORCH_CHECK(comp_ != nullptr)

    c10::List<at::Tensor> outputs;
    for (at::Tensor out : out_templates_) {
      outputs.push_back(at::empty_like(out));
    }

    // Adjust input memory formats
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

    // Adjust output memory formats
    auto out_mem_fmts = dict.at("out_mem_fmts").toIntList();
    TORCH_CHECK(outputs.size() == out_mem_fmts.size());
    for (int i = 0; i < outputs.size(); i++) {
      int fmt = out_mem_fmts[i];
      // These constants match the values in DimOrder in serializer.py
      // TODO: See if it's possible to use those directly.
      if (fmt == 1) {
        c10::IntArrayRef order = {0, 3, 1, 2};
        outputs.set(i, outputs.get(i).permute(order));
      } else if (fmt != 0) {
        throw std::exception();
        std::cerr << "Invalid mem_fmt" << std::endl;
      }
    }

    return c10::impl::toList(outputs);
  }

 private:
  // The following variables are modified by init() during execution,
  // and cannot be passed through the handles dictionary
  std::unique_ptr<torch::nnapi::bind::NnapiCompilation> comp_;
  c10::List<at::Tensor> out_templates_;
  at::Tensor ser_model_;
  mobile::Module shape_compute_module_;

  // Runs once per model initialization
  // Cannot be moved to compile(), because init() requires actual inputs
  void init(c10::IValue handle, c10::List<at::Tensor> inputs) {
    TORCH_CHECK(comp_ == nullptr);
    auto dict = handle.toGenericDict();

    // Load shape computation module
    std::stringstream ss;
    auto shape_ptr = dict.at("shape_compute_module").toString();
    ss.str(*shape_ptr);
    shape_compute_module_ = _load_for_mobile(ss);
    out_templates_ =
        shape_compute_module_.run_method("prepare", ser_model_, inputs)
            .toTensorList();

    // Create and initialize NnapiComilation object
    comp_ = std::make_unique<torch::nnapi::bind::NnapiCompilation>();
    auto weights = dict.at("weights").toTensorVector();
    comp_->init(ser_model_, weights);
  }
};

namespace {
constexpr auto backend_name = "nnapi";
static auto cls = torch::jit::backend<NnapiBackend>(backend_name);
} // namespace

} // namespace jit
} // namespace torch
