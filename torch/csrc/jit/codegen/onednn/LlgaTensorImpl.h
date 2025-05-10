#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/ir/ir.h>
#include <utility>

namespace torch::jit::fuser::onednn {

// Engine represents a device and its context. From the device kind, the engine
// knows how to generate code for the target device and what kind of device
// object to be expected. The device id ensures that there is a unique engine
// being created for each device. The device handle passed from PyTorch allows
// oneDNN Graph implementation to work on the device specified by PyTorch, which
// is currently CPU, so we only have one engine.
// Ref:
// https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onednn/source/graph/programming_model#engine
struct Engine {
  // CPU engine singleton
  static dnnl::engine& getEngine();
  Engine(const Engine&) = delete;
  void operator=(const Engine&) = delete;
};

// Stream is the logical abstraction for execution units. It is created on top
// of oneDNN Graph engine. A compiled oneDNN Graph partition is submitted to a
// stream for execution.
struct Stream {
  // CPU stream singleton
  static dnnl::stream& getStream();
  Stream(const Stream&) = delete;
  void operator=(const Stream&) = delete;
};

struct LlgaTensorDesc {
  using desc = dnnl::graph::logical_tensor;

  LlgaTensorDesc(
      size_t tid,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides,
      desc::data_type dtype,
      desc::property_type property_type)
      : tid_(tid),
        sizes_(std::move(sizes)),
        strides_(std::move(strides)),
        dtype_(dtype),
        property_type_(property_type),
        layout_type_(desc::layout_type::strided),
        layout_id_(-1) {}

  LlgaTensorDesc(const desc& t)
      : tid_(t.get_id()),
        sizes_(t.get_dims()),
        strides_({-1}),
        dtype_(t.get_data_type()),
        property_type_(t.get_property_type()),
        layout_type_(t.get_layout_type()),
        layout_id_(-1) {
    if (is_opaque()) {
      layout_id_ = t.get_layout_id();
    }
    if (is_strided()) {
      strides_ = t.get_strides();
    }
  }

  LlgaTensorDesc(const torch::jit::Value* v)
      : LlgaTensorDesc(
            v->unique(),
            {},
            {},
            desc::data_type::f32,
            get_property_type(v)) {
    if (v->type()->isSubtypeOf(TensorType::get())) {
      auto tt = v->type()->cast<TensorType>();

      if (tt->scalarType()) {
        dtype_ = getLlgaDataType(tt->scalarType().value());
      }

      auto sizes = tt->sizes();
      if (sizes.sizes()) {
        for (auto d : *sizes.sizes()) {
          sizes_.push_back(d.value_or(DNNL_GRAPH_UNKNOWN_DIM));
        }
      }

      auto strides = tt->strides();
      if (strides.sizes()) {
        for (auto d : *strides.sizes()) {
          strides_.push_back(d.value_or(DNNL_GRAPH_UNKNOWN_DIM));
        }
      }
    }
  }

  LlgaTensorDesc supplementTensorInfo(const at::Tensor& t) const;

  desc::data_type getLlgaDataType(at::ScalarType dt) const;

  at::ScalarType aten_scalar_type() const;

  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }

  const std::vector<int64_t>& strides() const {
    TORCH_CHECK(!is_opaque(), "Cannot get strides on opaque layout");
    return strides_;
  }

  size_t tid() const {
    return tid_;
  }

  LlgaTensorDesc tid(uint64_t new_id) const {
    auto ret = *this;
    ret.tid_ = new_id;
    return ret;
  }

  desc::data_type dtype() const {
    return dtype_;
  }

  LlgaTensorDesc dtype(desc::data_type new_dtype) const {
    return LlgaTensorDesc(tid_, sizes_, strides_, new_dtype, property_type_);
  }

  desc::layout_type layout_type() const {
    return layout_type_;
  }

  LlgaTensorDesc layout_type(desc::layout_type new_layout_type) {
    auto ret = *this;
    ret.layout_type_ = new_layout_type;
    return ret;
  }

  desc::property_type get_property_type(const torch::jit::Value* v) {
    switch (v->node()->kind()) {
      case prim::Constant:
        return desc::property_type::constant;
      default:
        return desc::property_type::variable;
    }
  }

  LlgaTensorDesc any() {
    return layout_type(desc::layout_type::any);
  }

  size_t storage_size() const {
    return logical_tensor().get_mem_size();
  }

  desc logical_tensor() const {
    if (is_dimensionality_unknown()) {
      return desc(
          tid_, dtype_, DNNL_GRAPH_UNKNOWN_NDIMS, layout_type_, property_type_);
    } else if (is_opaque()) {
      return desc(tid_, dtype_, sizes_, layout_id_, property_type_);
    } else if (is_any()) {
      return desc(tid_, dtype_, sizes_, layout_type_, property_type_);
    } else {
      return desc(tid_, dtype_, sizes_, strides_, property_type_);
    }
  }

  bool is_strided() const {
    return layout_type_ == desc::layout_type::strided;
  }

  bool is_any() const {
    return layout_type_ == desc::layout_type::any;
  }

  bool is_opaque() const {
    return layout_type_ == desc::layout_type::opaque;
  }

  bool operator==(const LlgaTensorDesc& desc) const {
    return tid_ == desc.tid_ && sizes_ == desc.sizes_ &&
        dtype_ == desc.dtype_ && layout_type_ == desc.layout_type_ &&
        ((is_opaque() && layout_id_ == desc.layout_id_) ||
         strides_ == desc.strides_);
  }

  bool operator!=(const LlgaTensorDesc& desc) const {
    return (tid_ != desc.tid_) || (sizes_ != desc.sizes_) ||
        (dtype_ != desc.dtype_) || (layout_type_ != desc.layout_type_) ||
        !((is_opaque() && (layout_id_ == desc.layout_id_)) ||
          (strides_ == desc.strides_));
  }

  static size_t hash(const LlgaTensorDesc& desc) {
    return c10::get_hash(
        desc.tid_,
        desc.sizes_,
        desc.dtype_,
        desc.layout_type_,
        desc.layout_id_);
  }

  void set_compute_inplace() {
    compute_inplace_ = true;
  }

  void set_input_tensor_index(size_t index) {
    input_tensor_index_ = index;
  }

  bool reuses_input_tensor() {
    return compute_inplace_;
  }

  size_t get_input_tensor_index() {
    return input_tensor_index_;
  }

 private:
  bool is_dimensionality_unknown() const {
    return sizes_.empty();
  }

  size_t tid_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  desc::data_type dtype_;
  desc::property_type property_type_;
  desc::layout_type layout_type_;
  size_t layout_id_;
  // If this is an output tensor, and querying the compiled partition would
  // determine that this tensor would reuse its input tensor, then
  // compute_inplace would be true, and input_tensor_index would be the index of
  // the corresponding input tensor in inputSpecs_ of the LlgaKernel object.
  bool compute_inplace_ = false;
  size_t input_tensor_index_{};
};

// Initially, oneDNN Graph also used to have blocked layout for tensors between
// partitions, and the LlgaTensorImpl wrapper helped us bypass guard checks.
// oneDNN Graph has switched over to using strided tensors between partitions,
// but this wrapper still helps us bypass guard checks because the strides of
// tensors between partitions would be different from the ones the guard is
// otherwise expecting.
struct TORCH_API LlgaTensorImpl : public c10::TensorImpl {
  LlgaTensorImpl(
      at::Storage&& storage,
      const caffe2::TypeMeta& data_type,
      const LlgaTensorDesc& desc);

  const LlgaTensorDesc& desc() const {
    return desc_;
  }

  static at::Tensor llga_to_aten_tensor(LlgaTensorImpl* llgaImpl);

 private:
  LlgaTensorDesc desc_;
};

at::Tensor empty_llga(
    const LlgaTensorDesc& desc,
    const c10::TensorOptions& options);

dnnl::graph::tensor llga_from_aten_tensor(const at::Tensor& tensor);

} // namespace torch::jit::fuser::onednn
