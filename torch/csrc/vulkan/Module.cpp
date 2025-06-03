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
#include <cstdint>
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

template <typename T>
std::uint32_t safe_cast_to_u32(T x) {
  if constexpr (std::is_unsigned_v<T>) {
    TORCH_CHECK(x <= std::numeric_limits<std::int32_t>::max());
  } else {
    TORCH_CHECK(
        x <= std::numeric_limits<std::int32_t>::max() &&
        x >= std::numeric_limits<int32_t>::min());
  }
  return static_cast<std::uint32_t>(x);
}

template <typename T>
at::native::vulkan::api::utils::uvec3 uvec3_from_vector(
    const std::vector<T>& vec,
    std::uint32_t default_element = 0) {
  TORCH_CHECK(
      vec.size() <= 3,
      "expected at-most-3-element vector but size was ",
      vec.size());
  std::uint32_t vec_0 = default_element;
  std::uint32_t vec_1 = default_element;
  std::uint32_t vec_2 = default_element;
  switch (vec.size()) {
    case 3:
      vec_2 = safe_cast_to_u32(vec[2]);
      [[fallthrough]];
    case 2:
      vec_1 = safe_cast_to_u32(vec[1]);
      [[fallthrough]];
    case 1:
      vec_0 = safe_cast_to_u32(vec[0]);
      [[fallthrough]];
    case 0:
      break;
  }
  return {vec_0, vec_1, vec_2};
}

template <size_t N>
at::native::vulkan::api::UniformParamsBuffer
vector_to_uniform_params_buffer_helper(
    at::native::vulkan::api::Context* context,
    const std::vector<std::int32_t>& vec) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vec.size() == N);
  std::array<std::int32_t, N> arr;
  std::copy(vec.begin(), vec.end(), arr.begin());
  return at::native::vulkan::api::UniformParamsBuffer(context, &arr);
}

at::native::vulkan::api::UniformParamsBuffer vector_to_uniform_params_buffer(
    at::native::vulkan::api::Context* context,
    const std::vector<std::int32_t>& vec) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!vec.empty());
#define VTUPB_CASE(N) \
  case N:             \
    return vector_to_uniform_params_buffer_helper<N>(context, vec)
  switch (vec.size()) {
    VTUPB_CASE(1);
    VTUPB_CASE(2);
    VTUPB_CASE(3);
    VTUPB_CASE(4);
    VTUPB_CASE(5);
    VTUPB_CASE(6);
    VTUPB_CASE(7);
    VTUPB_CASE(8);
    VTUPB_CASE(9);
    VTUPB_CASE(10);
    VTUPB_CASE(11);
    VTUPB_CASE(12);
    VTUPB_CASE(13);
    VTUPB_CASE(14);
    VTUPB_CASE(15);
    VTUPB_CASE(16);
    default:
      TORCH_CHECK(
          false,
          "compiled Vulkan shader received ",
          vec.size(),
          " non-Tensor arguments but only up to 16 are supported");
  }
#undef VTUPB_CASE
}
} // namespace

void initModule(PyObject* module) {
  using namespace at::native::vulkan;
  auto m = py::handle(module).cast<py::module>();
  py::class_<api::DynamicShaderInfo, std::shared_ptr<api::DynamicShaderInfo>>(
      m, "_vulkan_ShaderInfo")
      // NOTE: see the _compile_shader docstring in torch/vulkan/__init__.py for
      // the shader calling convention. IMPORTANT: DynamicShaderInfo.cpp depends
      // on this calling convention and must be changed if you change it! I
      // strongly recommend using the Vulkan validation layers when working on
      // this.

      .def(
          "__call__",
          [](api::DynamicShaderInfo& self,
             const py::args& args,
             const py::object& py_threads,
             const py::object& py_group_size) {
            TORCH_CHECK(
                !args.empty(),
                "must provide at least one argument to compiled Vulkan shader!");
            TORCH_CHECK(
                THPVariable_Check(args[0].ptr()),
                "first argument to compiled Vulkan shader must be a Tensor!");
            auto output_tensor = THPVariable_Unpack(args[0].ptr());
            auto& v_output_tensor = ops::convert(output_tensor);
            const auto output_storage_type = v_output_tensor.storage_type();
            auto threads =
                optional_vec_from_pyobject<std::uint32_t>(py_threads);
            TORCH_CHECK(
                !threads.has_value() || threads->size() < 4,
                "threads dimension must be less than 4");
            auto group_size =
                optional_vec_from_pyobject<std::uint32_t>(py_group_size);
            TORCH_CHECK(
                !group_size.has_value() || group_size->size() < 4,
                "group size dimension must be less than 4");

            auto* const context = api::context();

            api::PipelineBarrier pipeline_barrier{};
            std::vector<vTensor> input_args;
            std::vector<std::int32_t> int_constants;
            input_args.reserve(args.size() - 1);
            bool saw_non_tensor_arg = false;
            for (const auto ii : c10::irange(1, args.size())) {
              const auto& arg = args[ii];
              if (THPVariable_Check(arg.ptr())) {
                TORCH_CHECK(
                    !saw_non_tensor_arg,
                    "all non-Tensor arguments to compiled Vulkan shaders must come after Tensor arguments");
                const auto& v_input =
                    ops::convert(THPVariable_Unpack(args[ii].ptr()));
                TORCH_CHECK(
                    (v_input.storage_type() == api::StorageType::BUFFER) ==
                        (output_storage_type == api::StorageType::BUFFER),
                    "all Tensor arguments to compiled Vulkan shader must have the same storage type for now. output is ",
                    output_storage_type,
                    " but input ",
                    ii,
                    " is ",
                    v_input.storage_type());
                input_args.push_back(v_input);
              } else {
                saw_non_tensor_arg = true;
                TORCH_CHECK(
                    py::isinstance<py::int_>(arg),
                    "Only Tensor and int arguments to compiled Vulkan shaders are supported right now");
                const auto arg_value = arg.cast<std::int64_t>();
                TORCH_CHECK(
                    arg_value <= std::numeric_limits<std::int32_t>::max() &&
                        arg_value >= std::numeric_limits<int32_t>::min(),
                    "argument ",
                    arg_value,
                    " to compiled Vulkan shader does not fit in 32-bit integer");
                int_constants.push_back(static_cast<std::int32_t>(arg_value));
              }
            }
            const bool use_buffers =
                output_storage_type == api::StorageType::BUFFER;
            // TODO: no-gil support; we need to synchronize access to
            // self because we write to it if the GIL isn't doing it
            // for us.
            if (!self.layout_is_initialized()) {
              api::ShaderLayout::Signature layout;
              const auto actual_num_args = 1 /* the output */ +
                  input_args.size() + (int_constants.empty() ? 0 : 1);
              TORCH_CHECK_EQ(
                  self.get_expected_number_of_arguments(),
                  int(actual_num_args));
              layout.reserve(actual_num_args);
              layout.push_back(
                  use_buffers ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                              : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
              for (const auto _ : c10::irange(input_args.size())) {
                layout.push_back(
                    use_buffers ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
              }
              TORCH_CHECK(
                  use_buffers || int_constants.empty(),
                  "image calling convention to compiled Vulkan shader does not accept extra integer arguments!");
              if (!int_constants.empty()) {
                layout.push_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
              }
              self.set_layout(std::move(layout));
            }
            auto submit_job_buffer = [&](auto&... args) {
              // It is a hassle to set up push constants in the
              // backend, so we just make a uniform buffer for
              // now. TODO: set up and use push constants.

              const auto threads_arg = threads.has_value()
                  ? uvec3_from_vector(*threads, /* default_element = */ 1)
                  : uvec3_from_vector(
                        v_output_tensor.sizes(), /* default_element = */ 1);
              const auto group_size_arg = group_size.has_value()
                  ? uvec3_from_vector(*group_size)
                  : adaptive_work_group_size(
                        uvec3_from_vector(v_output_tensor.sizes()));
              const auto help_submit_job = [&](auto&... inner_args) {
                context->submit_compute_job(
                    self.shader_info(),
                    pipeline_barrier,
                    threads_arg,
                    group_size_arg,
                    VK_NULL_HANDLE,
                    v_output_tensor.buffer(
                        pipeline_barrier,
                        api::PipelineStage::COMPUTE,
                        api::MemoryAccessType::WRITE),
                    inner_args...);
              };
              if (int_constants.empty()) {
                help_submit_job(args.buffer(
                    pipeline_barrier, api::PipelineStage::COMPUTE)...);
              } else {
                api::UniformParamsBuffer params =
                    vector_to_uniform_params_buffer(context, int_constants);
                help_submit_job(
                    args.buffer(
                        pipeline_barrier, api::PipelineStage::COMPUTE)...,
                    params.buffer());
              }
            };
            auto submit_job_image = [&](auto&... args) {
              // TODO: this should be a push constant (or able to be
              // specified as one?) but we are matching existing shader
              // convention for now.
              struct Block {
                api::utils::uvec3 size;
              } block = {v_output_tensor.extents()};
              api::UniformParamsBuffer params(context, block);
              const auto threads_arg = threads.has_value()
                  ? uvec3_from_vector(*threads, /* default_element = */ 1)
                  : v_output_tensor.extents();
              const auto group_size_arg = group_size.has_value()
                  ? uvec3_from_vector(*group_size, /* default_element = */ 1)
                  : adaptive_work_group_size(v_output_tensor.extents());
              context->submit_compute_job(
                  self.shader_info(),
                  pipeline_barrier,
                  threads_arg,
                  group_size_arg,
                  VK_NULL_HANDLE,
                  v_output_tensor.image(
                      pipeline_barrier,
                      api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::WRITE),
                  args.image(pipeline_barrier, api::PipelineStage::COMPUTE)...,
                  params.buffer());
            };
            auto submit_job = [&](auto&... args) {
              if (use_buffers) {
                submit_job_buffer(args...);
              } else {
                submit_job_image(args...);
              }
            };
            switch (input_args.size()) {
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
      // TODO: make use_buffers a string or enum "calling convention" arg?
      [](const std::string& name,
         const std::string& source,
         const bool use_buffers) {
        return std::make_shared<api::DynamicShaderInfo>(
            api::compile_glsl(name, source, use_buffers));
      });
}
#else // USE_VULKAN
void initModule(PyObject* module) {}
#endif // USE_VULKAN

} // namespace torch::vulkan
