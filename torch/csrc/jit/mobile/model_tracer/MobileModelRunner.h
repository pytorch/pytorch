#pragma once

#include <mutex>
#include <sstream>

#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/script.h>

namespace torch {
namespace jit {
namespace mobile {

class MobileModelRunner {
  std::shared_ptr<torch::jit::mobile::Module> module_;

 public:
  explicit MobileModelRunner(std::string const& file_path) {
    module_ = std::make_shared<torch::jit::mobile::Module>(
        torch::jit::_load_for_mobile(file_path));
  }

  MobileModelRunner(
      std::string const& file_path,
      uint64_t module_load_options) {
    std::unordered_map<std::string, std::string> extra_files;
    module_ = std::make_shared<torch::jit::mobile::Module>(
        torch::jit::_load_for_mobile(
            file_path,
            at::Device(at::DeviceType::CPU, 0),
            extra_files,
            module_load_options));
  }

  MobileModelRunner(std::stringstream oss) {
    module_ = std::make_shared<torch::jit::mobile::Module>(
        torch::jit::_load_for_mobile(oss, at::Device(at::DeviceType::CPU, 0)));
  }

  /**
   * Returns true if the list of operators passed in has a Metal GPU operator,
   * and false otherwise.
   *
   */
  static bool set_has_metal_gpu_operators(std::set<std::string> const& op_list);

  /**
   * Fetches the set of root operators in the file "extra/mobile_info.json"
   * within the .ptl archive at location file_path.
   *
   * An exception is thrown if:
   *
   * 1. The file at file_path does not exist, or
   * 2. The contents of extra/mobile_info.json is not a JSON, or
   * 3. The file extra/mobile_info.json does not exist, or
   * 4. The JSON is malformed in some way and the operator list can not be
   * extracted correctly.
   *
   */
  static std::set<std::string> get_operators_from_mobile_info_json(
      std::string const& file_path);

  static std::vector<std::vector<at::IValue>> ivalue_to_bundled_inputs(
      const c10::IValue& bundled_inputs);

  static std::unordered_map<std::string, std::string>
  ivalue_to_bundled_inputs_map(const c10::IValue& bundled_inputs);

  /**
   * Fetches all the bundled inputs of the loaded mobile model.
   *
   * A bundled input itself is of type std::vector<at::IValue> and the
   * elements of this vector<> are the arguments that the "forward"
   * method of the model accepts. i.e. each of the at::IValue is a
   * single argument to the model's "forward" method.
   *
   * The outer vector holds a bundled input. For models with bundled
   * inputs, the outer most vector will have size > 0.
   */
  std::vector<std::vector<at::IValue>> get_all_bundled_inputs();

  /**
   * Fetches all the bundled inputs for all functions of the loaded mobile
   * model.
   *
   * The mapping is from 'function_names' eg 'forward' to bundled inputs for
   * that function
   *
   * A bundled input itself is of type std::vector<at::IValue> and the
   * elements of this vector<> are the arguments that the corresponding
   * method of the model accepts. i.e. each of the at::IValue in the entry
   * for forward is a single argument to the model's "forward" method.
   *
   * The outer vector of each value holds a bundled input. For models with
   * bundled inputs, the outer most vector will have size > 0.
   */
  std::unordered_map<std::string, std::vector<std::vector<at::IValue>>>
  get_many_functions_bundled_inputs();

  /**
   * Returns true if a model possesses get_bundled_inputs_functions_and_info()
   */
  bool has_new_style_bundled_inputs() const {
    return module_->find_method("get_bundled_inputs_functions_and_info") !=
        c10::nullopt;
  }

  /**
   * For each tensor in bundled inputs, call the user-provided function 'func'.
   */
  void for_each_tensor_in_bundled_inputs(
      std::function<void(const ::at::Tensor&)> const& func);

  /**
   * Get the root operators directly called by this model's Bytecode.
   */
  std::set<std::string> get_root_operators() {
    return torch::jit::mobile::_export_operator_list(*module_);
  }

  /**
   * Runs the model against all of the provided inputs using the model's
   * "forward" method. Returns an std::vector<at::IValue>, where each element
   * of the returned vector is one of the return values from calling forward().
   */
  std::vector<at::IValue> run_with_inputs(
      std::vector<std::vector<at::IValue>> const& bundled_inputs);

  /**
   * Runs the model against all of the provided inputs for all the specified
   * function. Returns an std::vector<at::IValue>, where each element
   * of the returned vector is one of the return values from calling the
   * method named "function_name" on this model.
   */
  std::vector<at::IValue> run_with_inputs(
      const std::string& function_name,
      std::vector<std::vector<at::IValue>> const& bundled_inputs) const;

  /**
   * Attempts to run all functions in the passed in list if they exist. All
   * funcs should require no args
   */
  void run_argless_functions(const std::vector<std::string>& functions);
};

} // namespace mobile
} // namespace jit
} // namespace torch
