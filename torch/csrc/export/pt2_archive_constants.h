#pragma once

#include <array>
#include <string_view>

namespace torch::_export::archive_spec {

#define FORALL_CONSTANTS(DO)                                                   \
  DO(ARCHIVE_ROOT_NAME, "package")                                             \
  /* Archive format */                                                         \
  DO(ARCHIVE_FORMAT_PATH, "archive_format")                                    \
  DO(ARCHIVE_FORMAT_VALUE, "pt2")                                              \
  /* Archive version */                                                        \
  DO(ARCHIVE_VERSION_PATH, "archive_version")                                  \
  DO(ARCHIVE_VERSION_VALUE, "0") /* Sep.4.2024: This is the initial version of \
                                    the PT2 Archive Spec */                    \
  /*                                                                           \
   * ######## Note on updating ARCHIVE_VERSION_VALUE ########                  \
   * When there is a BC breaking change to the PT2 Archive Spec,               \
   * e.g. deleting a folder, or changing the naming convention of the          \
   * following fields it would require bumping the ARCHIVE_VERSION_VALUE       \
   * Archive reader would need corresponding changes to support loading both   \
   * the current and older versions of the PT2 Archive.                        \
   */                                                                          \
  /* Model definitions */                                                      \
  DO(MODELS_DIR, "models/")                                                    \
  DO(MODELS_FILENAME_FORMAT, "models/{}.json") /* {model_name} */              \
  /* AOTInductor artifacts */                                                  \
  DO(AOTINDUCTOR_DIR, "data/aotinductor/")                                     \
  /* MTIA artifacts */                                                         \
  DO(MTIA_DIR, "data/mtia")                                                    \
  /* weights, including parameters and buffers */                              \
  DO(WEIGHTS_DIR, "data/weights/")                                             \
  DO(WEIGHT_FILENAME_PREFIX, "weight_")                                        \
  DO(WEIGHTS_PARAM_CONFIG_FORMAT, "data/weights/{}_model_param_config.json")   \
  /* constants, including tensor_constants, non-persistent buffers and script  \
   * objects */                                                                \
  DO(CONSTANTS_DIR, "data/constants/")                                         \
  DO(CONSTANTS_PARAM_CONFIG_FORMAT,                                            \
     "data/constants/{}_model_constants_config.json")                          \
  DO(TENSOR_CONSTANT_FILENAME_PREFIX, "tensor_")                               \
  DO(CUSTOM_OBJ_FILENAME_PREFIX, "custom_obj_")                                \
  /* example inputs */                                                         \
  DO(SAMPLE_INPUTS_DIR, "data/sample_inputs/")                                 \
  DO(SAMPLE_INPUTS_FILENAME_FORMAT,                                            \
     "data/sample_inputs/{}.pt") /* {model_name} */                            \
  /* extra folder */                                                           \
  DO(EXTRA_DIR, "extra/")                                                      \
  DO(MODULE_INFO_PATH, "extra/module_info.json")                               \
  /* xl_model_weights, this folder is used for storing per-feature-weights for \
   * remote net data in this folder is consume by Predictor, and is not        \
   * intended to be used by Sigmoid */                                         \
  DO(XL_MODEL_WEIGHTS_DIR, "xl_model_weights/")                                \
  DO(XL_MODEL_WEIGHTS_PARAM_CONFIG_PATH, "xl_model_weights/model_param_config")

#define DEFINE_GLOBAL(NAME, VALUE) \
  inline constexpr std::string_view NAME = VALUE;
FORALL_CONSTANTS(DEFINE_GLOBAL)
#undef DEFINE_GLOBAL

#define DEFINE_ENTRY(NAME, VALUE) std::pair(#NAME, VALUE),
inline constexpr std::array kAllConstants{FORALL_CONSTANTS(DEFINE_ENTRY)};
#undef DEFINE_ENTRY

#undef FORALL_CONSTANTS
} // namespace torch::_export::archive_spec
