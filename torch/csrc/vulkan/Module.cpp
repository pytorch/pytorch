#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <ATen/ATen.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/vulkan/impl/Common.h>
#include <ATen/native/vulkan/ops/Convert.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/mps/Module.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <memory>

#ifdef USE_VULKAN
#include <ATen/native/vulkan/api/DynamicShaderInfo.h>
#endif

namespace torch::vulkan {

// NOLINTNEXTLINE(*-c-arrays, *-global-variables)
static struct PyMethodDef _VulkanModule_methods[] = {{nullptr}};

PyMethodDef* python_functions() {
  return _VulkanModule_methods;
}

#ifdef USE_VULKAN
namespace {
template <typename T = uint64_t>
std::optional<std::vector<T>> optional_vec_from_pyobject(
    const py::object& py_value) {
  if (py_value.is_none()) {
    return std::nullopt;
  }
  if (py::isinstance<py::int_>(py_value)) {
    return std::vector({py_value.cast<T>()});
  }
  auto vec = py_value.cast<std::vector<T>>();
  TORCH_CHECK(vec.size() > 0 && vec.size() < 4);
  return vec;
}

} // namespace

void initModule(PyObject* module) {
  using namespace at::native::vulkan;
  auto m = py::handle(module).cast<py::module>();
  py::class_<api::DynamicShaderInfo, std::shared_ptr<api::DynamicShaderInfo>>(
      m, "_vulkan_ShaderInfo")
      // TODO: write a docstring explaining the following:
      // - we determine number of arguments by looking for layout(set lines in
      // the source (no includes allowed yet)
      // - shaders have the following calling convention:
      //   specialization constants must be used to set workgroup size, like so:
      //     layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id =
      //     2) in;
      //   binding 0 is a uniform restrict writeonly image3D where the output is
      //   to be written bindings 1-N (where N is the arity of the op) are
      //   uniform sampler3D objects binding N+1 is a uniform Block containing a
      //   single ivec3 indicating the operation size. (This should probably be
      //   a push constant (or specialization constant as done in the ExecuTorch
      //   backend???), but currently we are just matching existing shader
      //   convention.)
      //
      //   IMPORTANT: DynamicShaderInfo.cpp depends on this calling
      //   convention and must be changed! I strongly recommend using
      //   the Vulkan validation layers when working on this.

      .def(
          "__call__",
          [](api::DynamicShaderInfo& self,
             const py::args& args,
             const py::object& py_threads,
             const py::object& py_group_size) {
            TORCH_CHECK(
                args.size() > 0,
                "must provide at least one argument to compiled Vulkan shader!");
            TORCH_CHECK(
                THPVariable_Check(args[0].ptr()),
                "first argument to compiled Vulkan shader must be a Tensor!");
            auto output_tensor = THPVariable_Unpack(args[0].ptr());
            auto& v_output_tensor = ops::convert(output_tensor);
            // TODO: this should be a push constant (or able to be
            // specified as one?) but we are matching existing shader
            // convention for now.
            struct Block {
              api::utils::uvec3 size;
            } block;
            // TODO: grab output (arg0) and use to set operation extents.
            block.size = v_output_tensor.extents();
            auto threads =
                optional_vec_from_pyobject<std::uint32_t>(py_threads);
            TORCH_CHECK(
                !threads.has_value() || threads->size() == 3,
                "threads dimension must be 3");
            auto group_size =
                optional_vec_from_pyobject<std::uint32_t>(py_group_size);
            TORCH_CHECK(
                !group_size.has_value() || group_size->size() == 3,
                "group size dimension must be 3");

            auto* const context = api::context();
            api::UniformParamsBuffer params(context, block);

            api::PipelineBarrier pipeline_barrier;
            std::vector<vTensor> input_args;
            input_args.reserve(args.size() - 1);
            for (const auto ii : c10::irange(1, args.size())) {
              TORCH_CHECK(
                  THPVariable_Check(args[ii].ptr()),
                  "non-Tensor arguments to compiled Vulkan shaders are not supported right now");
              input_args.push_back(
                  ops::convert(THPVariable_Unpack(args[ii].ptr())));
            }
            auto submit_job = [&](auto&... args) {
              context->submit_compute_job(
                  self.shader_info(),
                  pipeline_barrier,
                  threads.has_value()
                      ? api::utils::
                            uvec3{(*threads)[0], (*threads)[1], (*threads)[2]}
                      : v_output_tensor.extents(),
                  group_size.has_value()
                      ? api::utils::
                            uvec3{(*group_size)[0], (*group_size)[1], (*group_size)[2]}
                      : adaptive_work_group_size(v_output_tensor.extents()),
                  VK_NULL_HANDLE,
                  v_output_tensor.image(
                      pipeline_barrier,
                      api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::WRITE),
                  args.image(pipeline_barrier, api::PipelineStage::COMPUTE)...,
                  params.buffer());
            };
            switch (args.size() - 1) {
              case 0:
                submit_job();
                break;
              case 1:
                submit_job(input_args[0]);
                break;
              case 2:
                submit_job(input_args[0], input_args[1]);
                break;
              case 3:
                submit_job(input_args[0], input_args[1], input_args[2]);
                break;
              case 4:
                submit_job(
                    input_args[0], input_args[1], input_args[2], input_args[3]);
                break;
              default:
                TORCH_CHECK(
                    false,
                    "Compiled Vulkan shaders may only have up to 4 input arguments for now");
            }
          },
          py::kw_only(),
          py::arg("threads") = py::none(),
          py::arg("group_size") = py::none());
  m.def(
      "_vulkan_compileShader",
      [](const std::string& name, const std::string& source) {
        return std::make_shared<api::DynamicShaderInfo>(
            api::compile_glsl(name, source));
      });
}
#endif /* USE_VULKAN */

} // namespace torch::vulkan
