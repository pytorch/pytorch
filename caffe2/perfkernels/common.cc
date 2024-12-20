#include "common.h"

#include <stdlib.h>
#include <string.h>

namespace caffe2::perfkernels {
namespace {
std::optional<SelectedIsa> getSelectedIsaFromEnv() {
  const char* env = getenv("CAFFE2_PERFKERNELS_SELECT_ISA");
  if (env == nullptr) {
    return std::nullopt;
  }
  if (strcmp(env, "avx2") == 0 || strcmp(env, "AVX2") == 0) {
    return SelectedIsa::avx2;
  }
  if (strcmp(env, "avx512") == 0 || strcmp(env, "AVX512") == 0) {
    return SelectedIsa::avx512;
  }
  if (strcmp(env, "avx2_fma") == 0 || strcmp(env, "AVX2_FMA") == 0) {
    return SelectedIsa::avx2_fma;
  }
  return std::nullopt;
}

std::optional<SelectedIsa>& getIsaStorage() {
  static std::optional<SelectedIsa> isa = getSelectedIsaFromEnv();
  return isa;
}
} // namespace

std::optional<SelectedIsa> getSelectedIsa() {
  return getIsaStorage();
}

SelectedIsa setIsa(SelectedIsa isa) {
  auto& storage = getIsaStorage();
  auto old_isa = storage.value_or(SelectedIsa::undefined);
  storage = isa;
  return old_isa;
}
} // namespace caffe2::perfkernels
