#include "OpenRegSerialization.h"

namespace c10::openreg {
struct OpenRegBackendMeta : public c10::BackendMeta {
  OpenRegBackendMeta(int version_number, int format_number)
      : version_number_(version_number), format_number_(format_number) {}

  int version_number_{-1};
  int format_number_{-1};
};

void for_serialization(
    const at::Tensor& t,
    std::unordered_map<std::string, bool>& m) {
  auto meta_ptr = t.unsafeGetTensorImpl()->get_backend_meta();

  if (meta_ptr != nullptr) {
    auto o_meta_ptr = dynamic_cast<OpenRegBackendMeta*>(meta_ptr);
    if (o_meta_ptr->version_number_ == 1) {
      m["version_number"] = true;
    }
    if (o_meta_ptr->format_number_ == 29) {
      m["format_number"] = true;
    }
  }
}

void for_deserialization(
    const at::Tensor& t,
    std::unordered_map<std::string, bool>& m) {
  int version_number{-1};
  int format_number{-1};

  if (m.find("version_number") != m.end()) {
    version_number = 1;
  }
  if (m.find("format_number") != m.end()) {
    format_number = 29;
  }

  c10::intrusive_ptr<c10::BackendMeta> meta{std::unique_ptr<c10::BackendMeta>(
      new OpenRegBackendMeta(version_number, format_number))};
  t.unsafeGetTensorImpl()->set_backend_meta(meta);
}

REGISTER_PRIVATEUSE1_SERIALIZATION(&for_serialization, &for_deserialization)

} // namespace c10::openreg
