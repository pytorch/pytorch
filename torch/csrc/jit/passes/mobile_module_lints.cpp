#include <torch/csrc/jit/passes/mobile_module_lints.h>

namespace torch {
namespace jit {

static std::string GetLintNameString(ModuleLintCode lint_code) {
    std::string lint_name_str;
    switch (lint_code) {
        case ModuleLintCode::ML_BUNDLED_INPUT:
            lint_name_str = "ML_BUNDLED_INPUT";
            break;
        default:
            lint_name_str = "NONE";

    }
    return lint_name_str;
}

static std::string GetLintDescriptionString(ModuleLintCode lint_code) {
    std::string lint_str;
    switch (lint_code) {
        case ModuleLintCode::ML_BUNDLED_INPUT:
            lint_str = "Please add bundled inputs before saving the module using "
            "torch.utils.bundled_inputs.augment_model_with_bundled_inputs.";
            break;
        default:
            lint_str = "Lint code is not supported";

    }
    return lint_str;
}


static void checkBundledInputs(std::map<std::string, std::string>& lints, const script::Module& module) {
    auto get_method = module.find_method("get_all_bundled_inputs");
    if (!get_method) {
        lints[GetLintNameString(ModuleLintCode::ML_BUNDLED_INPUT)] = GetLintDescriptionString(ModuleLintCode::ML_BUNDLED_INPUT);
    }
}

std::map<std::string, std::string> GenerateModuleLints(const jit::script::Module& module) {
    std::map<std::string, std::string> lint_map;
    checkBundledInputs(lint_map, module);
    return lint_map;
}

} // namespace jit
} // namespace torch
