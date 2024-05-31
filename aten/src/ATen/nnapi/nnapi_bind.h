#ifndef NNAPI_BIND_H_
#define NNAPI_BIND_H_

#include <vector>

#include <ATen/ATen.h>
#include <torch/custom_class.h>

#include <ATen/nnapi/nnapi_wrapper.h>

namespace torch {
namespace nnapi {
namespace bind {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TORCH_API extern nnapi_wrapper* nnapi;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TORCH_API extern nnapi_wrapper* check_nnapi;

#define MAKE_SMART_PTR(type) \
  struct type ## Freer { \
    void operator()(ANeuralNetworks ## type * obj) { \
      if (!nnapi) { /* obj must be null. */ return; } \
      nnapi-> type ## _free(obj); \
    } \
  }; \
  typedef std::unique_ptr<ANeuralNetworks ## type, type ## Freer> type ## Ptr;

MAKE_SMART_PTR(Model)
MAKE_SMART_PTR(Compilation)
MAKE_SMART_PTR(Execution)

#undef MAKE_SMART_PTR

struct NnapiCompilation : torch::jit::CustomClassHolder {
    NnapiCompilation() = default;
    ~NnapiCompilation() override = default;

    // only necessary for older models that still call init()
    TORCH_API void init(
      at::Tensor serialized_model_tensor,
      std::vector<at::Tensor> parameter_buffers
    );

    TORCH_API void init2(
      at::Tensor serialized_model_tensor,
      const std::vector<at::Tensor>& parameter_buffers,
      int64_t compilation_preference,
      bool relax_f32_to_f16
    );


    TORCH_API void run(std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs);
    static void get_operand_type(const at::Tensor& t, ANeuralNetworksOperandType* operand, std::vector<uint32_t>* dims);

    ModelPtr model_;
    CompilationPtr compilation_;
    int32_t num_inputs_ {};
    int32_t num_outputs_ {};
};

} // namespace bind
} // namespace nnapi
} // namespace torch

#endif // NNAPI_BIND_H_
