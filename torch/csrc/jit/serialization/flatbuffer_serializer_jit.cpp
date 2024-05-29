#include <torch/csrc/jit/serialization/flatbuffer_serializer_jit.h>

#ifdef FLATBUFFERS_VERSION_MAJOR
#error "flatbuffer_serializer_jit.h must not include any flatbuffers headers"
#endif // FLATBUFFERS_VERSION_MAJOR

#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/operator_upgraders/upgraders_entry.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>
#include <torch/csrc/jit/serialization/import.h>

namespace torch::jit {

bool register_flatbuffer_all() {
  return true;
}

} // namespace torch::jit
