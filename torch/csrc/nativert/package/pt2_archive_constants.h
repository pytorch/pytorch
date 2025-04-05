#pragma once

#include <array>
#include <string_view>

namespace torch::nativert {

namespace archive_spec {

#define FORALL_COSTANTS(_)                                                     \
  _(ARCHIVE_ROOT_NAME, "package")                                              \
  /* Archive format */                                                         \
  _(ARCHIVE_FORMAT_PATH, "archive_format")                                     \
  _(ARCHIVE_FORMAT_VALUE, "pt2")                                               \
  /* Archive version */                                                        \
  _(ARCHIVE_VERSION_PATH, "archive_version")                                   \
  _(ARCHIVE_VERSION_VALUE,                                                     \
    "0") /* Sep.4.2024: This is the initial version of the PT2 Archive Spec */ \
  /*                                                                           \
   * ######## Note on updating ARCHIVE_VERSION_VALUE ########                  \
   * When there is a BC breaking change to the PT2 Archive Spec,               \
   * e.g. deleting a folder, or changing the naming convention of the          \
   * following fields it would require bumping the ARCHIVE_VERSION_VALUE       \
   * Archive reader would need corresponding changes to support loading both   \
   * the current and older versions of the PT2 Archive.                        \
   */                                                                          \
  /* Model definitions */                                                      \
  _(MODELS_DIR, "models/")                                                     \
  _(MODELS_FILENAME_FORMAT, "models/{}.json") /* {model_name} */               \
  /* AOTInductor artifacts */                                                  \
  _(AOTINDUCTOR_DIR, "data/aotinductor/")                                      \
  /* MTIA artifacts */                                                         \
  _(MTIA_DIR, "data/mtia")                                                     \
  /* weights, including parameters and buffers */                              \
  _(WEIGHTS_DIR, "data/weights/")                                              \
  _(WEIGHT_FILENAME_PREFIX, "weight_")                                         \
  /* constants, including tensor_constants, non-persistent buffers and script  \
   * objects */                                                                \
  _(CONSTANTS_DIR, "data/constants/")                                          \
  _(TENSOR_CONSTANT_FILENAME_PREFIX, "tensor_")                                \
  _(CUSTOM_OBJ_FILENAME_PREFIX, "custom_obj_")                                 \
  /* example inputs */                                                         \
  _(SAMPLE_INPUTS_DIR, "data/sample_inputs/")                                  \
  _(SAMPLE_INPUTS_FILENAME_FORMAT,                                             \
    "data/sample_inputs/{}.pt") /* {model_name} */                             \
  /* extra folder */                                                           \
  _(EXTRA_DIR, "extra/")                                                       \
  _(MODULE_INFO_PATH, "extra/module_info.json")                                \
  /* xl_model_weights, this folder is used for storing per-feature-weights for \
   * remote net data in this folder is consume by Predictor, and is not        \
   * intended to be used by Sigmoid */                                         \
  _(XL_MODEL_WEIGHTS_DIR, "xl_model_weights/")                                 \
  _(XL_MODEL_WEIGHTS_PARAM_CONFIG_PATH, "xl_model_weights/model_param_config")

#define DEFINE_GLOBAL(NAME, VALUE) \
  inline constexpr std::string_view NAME = VALUE;

#define DEFINE_ENTRY(NAME, VALUE) std::pair(#NAME, VALUE),

FORALL_COSTANTS(DEFINE_GLOBAL)

inline constexpr std::array kAllConstants{FORALL_COSTANTS(DEFINE_ENTRY)};

#undef DEFINE_ENTRY
#undef DEFINE_GLOBAL
#undef FORALL_COSTANTS
} // namespace archive_spec
} // namespace torch::nativert
