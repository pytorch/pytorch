#include "OpenRegSerialization.h"

namespace c10::openreg {
struct OpenRegBackendMeta : public c10::BackendMeta {
  OpenRegBackendMeta(int version_number, int format_number)
      : version_number_(version_number), format_number_(format_number) {}

  // Carry OpenReg metadata onto FakeTensors, tagging the copy so tests can tell
  // it went through clone_for_fake() rather than a raw clone().
  c10::intrusive_ptr<c10::BackendMeta> clone_for_fake(
      const c10::intrusive_ptr<c10::BackendMeta>& ptr
      [[maybe_unused]]) const override {
    auto cloned = c10::make_intrusive<OpenRegBackendMeta>(
        version_number_, format_number_);
    cloned->fake_clone_ = true;
    return cloned;
  }

  int version_number_{-1};
  int format_number_{-1};
  bool fake_clone_{false};
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
    // Set by clone_for_fake(); lets tests confirm the fakeification path.
    if (o_meta_ptr->fake_clone_) {
      m["fake_clone"] = true;
    }
  }
}

void for_deserialization(
    const at::Tensor& t,
    std::unordered_map<std::string, bool>& m) {
  int version_number{-1};
  int format_number{-1};

  if (m.contains("version_number")) {
    version_number = 1;
  }
  if (m.contains("format_number")) {
    format_number = 29;
  }

  c10::intrusive_ptr<c10::BackendMeta> meta{std::unique_ptr<c10::BackendMeta>(
      new OpenRegBackendMeta(version_number, format_number))};
  t.unsafeGetTensorImpl()->set_backend_meta(meta);
}

REGISTER_PRIVATEUSE1_SERIALIZATION(&for_serialization, &for_deserialization)

} // namespace c10::openreg
