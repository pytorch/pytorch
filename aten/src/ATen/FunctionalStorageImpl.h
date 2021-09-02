#pragma once

#include <ATen/Tensor.h>

namespace at {
namespace functionalization {

// See Note [Functionalization Pass In Core]

struct ViewMeta {
  ViewMeta(
          std::function<Tensor(const Tensor&, int64_t)> forward,
          std::function<Tensor(const Tensor&, const Tensor&, int64_t)> reverse,
          int64_t out_idx = 0) :
      forward_fn(forward),
      reverse_fn(reverse),
      out_index(out_idx)
      {}

  std::function<Tensor(const Tensor&, int64_t)> forward_fn;
  std::function<Tensor(const Tensor&, const Tensor&, int64_t)> reverse_fn;
  // See Note [out_idx in ViewMeta]
  int64_t out_index;
};

class Alias {
  public:
    struct Update {
        const at::Tensor new_val;
        std::vector<ViewMeta> view_metas;
    };
    explicit Alias(const at::Tensor& base);
    const at::Tensor& base() const;
    size_t generation() const { return generation_; }
    void add_update(const at::Tensor& updated_val, std::vector<ViewMeta> metas);
    void apply_update(const Update& update);
    Tensor sync_update_operations();
  private:
    at::Tensor base_;
    std::vector<Update> updates_;
    size_t generation_ = 0;
};

struct C10_API FunctionalStorageImpl : public c10::StorageImpl {
  explicit FunctionalStorageImpl(c10::Device device, int64_t numel, caffe2::TypeMeta dtype);

  bool maybe_add_update(const Tensor& updated_val, std::vector<ViewMeta>& view_metas);
  Tensor sync_update_operations();
  void set_alias(const Tensor& alias);
  bool is_aliased() const;
  size_t generation() const;

 private:
  std::unique_ptr<at::functionalization::Alias> alias_ = nullptr;
};

} // namespace functionalization
} // namespace at
