#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/detail/Attr.h>
#include <ATen/native/mkldnn/detail/Utils.h>
#include <dnnl.h>
#include <dnnl.hpp>

#include <c10/macros/Export.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace at::native::mkldnn {

namespace utils {

// Ported from ideep::utils::allocator.
class allocator {
 public:
  constexpr static size_t tensor_memalignment = 4096;

  static char* malloc(size_t size) {
    void* ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, tensor_memalignment);
    int rc = ((ptr) ? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, tensor_memalignment, size);
#endif /* _WIN32 */
    return (rc == 0) ? (char*)ptr : nullptr;
  }

  static void free(void* p) {
#ifdef _WIN32
    _aligned_free((void*)p);
#else
    ::free((void*)p);
#endif /* _WIN32 */
  }
};

} // namespace utils

// Ported from ideep::engine.
/// cpu execution engine only.
struct Engine : public dnnl::engine {
  friend class ITensor;

  /// Singleton CPU engine for all primitives
  static TORCH_API Engine& cpu_engine();

  Engine(kind akind = kind::cpu, size_t index = 0)
      : dnnl::engine(akind, index),
        malloc(utils::allocator::malloc),
        free(utils::allocator::free) {}

  void set_allocator(
      const std::function<void*(size_t)>& malloc,
      const std::function<void(void*)>& free) {
    this->malloc = malloc;
    this->free = free;
  }

 private:
  std::function<void*(size_t)> malloc;
  std::function<void(void*)> free;
};

// Ported from ideep::stream.
/// A default stream
struct IStream : public dnnl::stream {
  static dnnl::stream& default_stream() {
    static dnnl::stream s(Engine::cpu_engine());
    return s;
  }
};

// Ported from ideep::tensor.
class ITensor : public dnnl::memory {
 public:
  using dim_t = dnnl_dim_t;
  using dims_t = dnnl_dims_t;
  using format_kind_t = memory::format_kind;

  struct desc : public memory::desc {
    friend class ITensor;

    // avoid conflicts with function desc::dims() and desc::data_type()
    using dims = typename memory::dims;
    using data_type = typename memory::data_type;

    desc() : memory::desc(){};

    // copy ctor
    desc(const desc& adesc) : memory::desc(adesc) {
      set_g(adesc.g());
    };

    // supplement group info for memory::desc
    desc(const memory::desc& adesc, dim groups = 1) : memory::desc(adesc) {
      set_g(groups);
    };

    desc& operator=(const desc& adesc) {
      memory::desc::operator=(adesc);
      set_g(adesc.g());
      return *this;
    }

    desc(const dnnl_memory_desc_t& adata) : memory::desc(adata){};

    desc(const dims& adims, data_type adata_type, format_tag aformat_tag)
        : memory::desc(adims, adata_type, aformat_tag) {
      set_g(1);
    }

    desc(const dims& adims, data_type adata_type)
        : desc(adims, adata_type, get_default_format(adims)) {}

    desc(const dims& adims, data_type adata_type, const dims& astrides)
        : memory::desc(adims, adata_type, astrides) {
      set_g(1);
    }

    // Construct a memory descriptor from a binary blob.
    // The API is available since oneDNN v3.4.1
    desc(const std::vector<uint8_t> &blob) : memory::desc(get_dnnl_blob(blob)) {
      // The groups is serialized into the last sizeof(int) item
      // We use little endian encoding here for the groups_bytes
      std::vector<uint8_t> groups_bytes(blob.end() - sizeof(int), blob.end());
      int groups = *reinterpret_cast<int*>(groups_bytes.data());
      set_g(groups);
    }

    void to_bytes(utils::bytestring& bytes) const {
      utils::to_bytes(bytes, get_data_type());
      utils::to_bytes(bytes, format_kind());
      utils::to_bytes(bytes, get_submemory_offset());

      auto paddim = get_padded_dims();
      auto padoff = get_padded_offsets();

      dim_t *c_dims = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_dims, &c_dims);

      for (int i = 0; i < get_internal_ndims(); i++) {
        utils::to_bytes(bytes, c_dims[i]);
        utils::to_bytes(bytes, paddim[i]);
        utils::to_bytes(bytes, padoff[i]);
      }

      if (is_blocking_desc()) {
        dim_t *c_strides = nullptr;
        dnnl_memory_desc_query(get(), dnnl_query_strides, &c_strides);
        for (int i = 0; i < get_internal_ndims(); i++) {
          utils::to_bytes(bytes, c_strides[i]);
        }
        for (int i = 0; i < get_inner_nblks(); i++) {
          utils::to_bytes(bytes, get_inner_idxs()[i]);
          utils::to_bytes(bytes, get_inner_blks()[i]);
        }
      }
    }

    /// public ndims
    inline int get_ndims() const {
      return is_grouped() ? get_internal_ndims() - 1 : get_internal_ndims();
    }

    /// Return size of specified dimension
    inline dim_t get_dim(int index) const {
      dim_t *c_dims = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_dims, &c_dims);
      int internal_ndims = get_internal_ndims();
      if (!is_grouped()) {
        if (index < 0 || index >= internal_ndims)
          return static_cast<dim_t>(0);
        return c_dims[index];
      } else {
        if (index < 0 || index >= internal_ndims - 1)
          return static_cast<dim_t>(0);
        return index == 0 ?
            c_dims[0] * c_dims[1] :
            c_dims[index + 1];
      }
    }

    /// Returns dimension vector
    inline dims get_dims() const {
      dim_t *c_dims = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_dims, &c_dims);
      int internal_ndims = get_internal_ndims();
      if (!is_grouped()) {
        return dims(c_dims, c_dims + internal_ndims);
      } else {
        auto ret = dims(c_dims + 1, c_dims + internal_ndims);
        ret[0] *= c_dims[0]; // g == data.dims[0]
        return ret;
      }
    }

    inline dims get_strides() const {
#ifndef NDEBUG
      dnnl::error::wrap_c_api(
          (is_plain()) ? dnnl_success : dnnl_invalid_arguments,
          "Call to_public() before get_strides()");
#endif
      dim_t *strides = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_strides, &strides);
      if (!is_grouped()) {
        return dims(strides, strides + get_internal_ndims());
      } else {
        auto ret = dims(strides + 1, strides + get_internal_ndims());
        ret[0] = std::min(strides[0], strides[1]);
        return ret;
      }
    }

    /** returns the number of elements including padding if \param with_padding
     * is true, and the number of data elements otherwise */
    inline dim_t nelems(bool with_padding = false) const {
      if (is_zero())
        return 0;
      dim_t *c_dims = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_dims, &c_dims);
      dim_t *c_padded_dims = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_padded_dims, &c_padded_dims);
      auto dims = with_padding ? c_padded_dims : c_dims;
      return std::accumulate(
          dims, dims + get_internal_ndims(), 1, std::multiplies<dim_t>());
    }

    inline bool is_plain() const {
      return is_blocking_desc() && get_inner_nblks() == 0;
    };

    inline bool is_opaque() const {
      return is_opaque_desc();
    }

    inline bool is_default() const {
      if (!is_plain())
        return false;

      dim_t *strides = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_strides, &strides);
      for (int i = 0; i < get_internal_ndims() - 1; i++) {
        if (strides[i] < strides[i + 1]) {
          return false;
        }
      }
      return true;
    }

    // The logic of this function differs from PyTorch
    // It may cause error in edge cases.
    // Avoid using it to determine memory layout in PyTorch.
    inline bool is_nhwc() const {
      if (!is_plain() || get_internal_ndims() != 4) return false;
      dim_t *dims = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_dims, &dims);
      dim_t *strides = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_strides, &strides);
      const auto n = 0, c = 1, h = 2, w = 3;
      return strides[n] == dims[h] * dims[w] * dims[c]
          && strides[h] == dims[w] * dims[c]
          && strides[w] == dims[c]
          && strides[c] == 1;
    };

    // The logic of this function differs from PyTorch
    // It may cause error in edge cases.
    // Avoid using it to determine memory layout in PyTorch.
    inline bool is_ndhwc() const {
      if (!is_plain() || get_internal_ndims() != 5) return false;
      dim_t *dims = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_dims, &dims);
      dim_t *strides = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_strides, &strides);
      const auto n = 0, c = 1, d =2, h = 3, w = 4;
      return strides[n] == dims[d] * dims[h] * dims[w] * dims[c]
          && strides[d] == dims[h] * dims[w] * dims[c]
          && strides[h] == dims[w] * dims[c]
          && strides[w] == dims[c]
          && strides[c] == 1;
    }

    inline bool is_channels_last() const {
      auto ndims = get_internal_ndims();
      if (!is_plain() ||
          !(ndims == 4 || ndims == 5 || ndims == 3))
        return false;
      dim_t *dims = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_dims, &dims);
      dim_t *strides = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_strides, &strides);
      if (ndims == 4) {
        const auto n = 0, c = 1, h = 2, w = 3;
        return strides[n] == dims[h] * dims[w] * dims[c] &&
            strides[h] == dims[w] * dims[c] && strides[w] == dims[c] &&
            strides[c] == 1;
      } else if (ndims == 5) {
        const auto n = 0, c = 1, d = 2, h = 3, w = 4;
        return strides[n] == dims[d] * dims[h] * dims[w] * dims[c] &&
            strides[d] == dims[h] * dims[w] * dims[c] &&
            strides[h] == dims[w] * dims[c] && strides[w] == dims[c] &&
            strides[c] == 1;
      } else {
        const auto n = 0, c = 1, w = 2;
        return strides[n] == dims[w] * dims[c] && strides[w] == dims[c] &&
            strides[c] == 1;
      }
    };

    // workaround for issue intel/mkl-dnn#588
    bool is_4c_blocked() {
      return get_inner_nblks() == 1 && get_inner_idxs()[0] == 1 &&
          get_inner_blks()[0] == 4;
    }

    desc to_format(format_tag aformat_tag) const {
      auto ret = desc(get_internal_dims(), get_data_type(), aformat_tag);
      ret.set_g(g());
      return ret;
    }

    desc to_format_any() const {
      auto ret = desc(get_internal_dims(), get_data_type(), format_tag::any);
      ret.set_g(g());
      return ret;
    }

    desc to_default_format() const {
      auto ret = desc(get_internal_dims(), get_data_type());
      ret.set_g(g());
      return ret;
    }

    desc clone() const {
      return desc(*this);
    }

    desc to_type(data_type atype, Engine aengine = Engine::cpu_engine()) const {
      // We cannot change data type of a desc directly.
      // Case 1: Return a copy of this if no change is needed.
      if (atype == get_data_type()) return clone();
      // Case 2: For desc of plain layout, we can create a new desc with strides
      if (is_plain()) {
        desc ret(memory::desc(get_internal_dims(), atype, memory::desc::get_strides()));
        ret.set_g(g());
        return ret;
      }
      // Case 3: For blocked layout, we cannot create a new desc with strides
      // Use binary primitive desc to query a desc so that we can
      // change the data type and keep the layout
      auto& src0_desc = *this;
      auto src1_desc = desc(get_internal_dims(), atype, format_tag::any);
      auto dst_desc = desc(get_internal_dims(), atype, format_tag::any);
      auto pd = dnnl::binary::primitive_desc(
          aengine, dnnl::algorithm::binary_add, src0_desc, src1_desc, dst_desc);
      auto ret = desc(pd.dst_desc());
      ret.set_g(g());
      return ret;
    }

    desc to_grouped(int groups, bool is_deconv = false) const {
      auto grouped_dims = utils::group_dims(get_internal_dims(), groups);
      // preserve tag `any` otherwise use plain format tag since we don't know the exact original tag
      format_tag f_tag = get_default_format(grouped_dims);
      if (get_format_kind() == format_kind::any) {
        f_tag = format_tag::any;
      }
      auto grouped_desc = desc(grouped_dims, get_data_type(), f_tag);
      grouped_desc.set_g(groups);
      return grouped_desc;
    }

    bool has_same_shape_as(const desc& that) const {
      if (get_internal_ndims() != that.get_internal_ndims())
        return false;
      dim_t *this_dims = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_dims, &this_dims);
      dim_t *that_dims = nullptr;
      dnnl_memory_desc_query(that.get(), dnnl_query_dims, &that_dims);
      return utils::array_cmp(this_dims, that_dims, get_internal_ndims());
    }

    // to be replaced with memory_desc_permute_axes in DNNL v1.3
    desc permute(const std::vector<int>& permute_axes = {}) const {
      if (get_internal_ndims() <= 1) {
        return clone();
      }
      auto perms = permute_axes;
      if (perms.empty()) {
        // If perms is empty, we need to init it.
        perms.resize(get_internal_ndims());
        std::iota(perms.rbegin(), perms.rend(), 0);
      }
      std::vector<int> expected_perms(perms.size(), -1);
      for (size_t i = 0; i < perms.size(); i++) {
        // The permute axis has different semantic between PyTorch and oneDNN
        // Map the permute axis from PyTorch to oneDNN.
        size_t new_shape_idx = i;
        size_t org_shape_idx = perms[i];
        expected_perms[org_shape_idx] = static_cast<int>(new_shape_idx);
      }
      auto permuted_md = memory::desc::permute_axes(expected_perms);
      auto ret = desc(permuted_md);
      ret.set_g(g());
      return ret;
    }

    desc transpose(dim dim0, dim dim1) const {
      std::vector<int> axes(get_internal_ndims());
      std::iota(axes.begin(), axes.end(), 0);
      std::swap(axes[dim0], axes[dim1]);
      return permute(axes);
    }

    // Returns a binary blob associated with the given memory descriptor
    // API available since oneDNN v3.4.1
    std::vector<uint8_t> get_blob() {
      std::vector<uint8_t> out_blob = memory::desc::get_blob();
      
      int groups = g();
      // We need to add the groups info into the serialized ideep desc.
      // For example, a 4D FWK weight with groups > 1 will become a 5D oneDNN md,
      // while FWK still needs it to be a 4D tensor.
      // ideep desc will wrap the 5D oneDNN md back to a 4D ideep desc
      // when interacting with the FWK.
      // We use little endian encoding here when serializing groups
      out_blob.insert(out_blob.end(), (uint8_t*)&groups, (uint8_t*)&groups + sizeof(int));
      
      return out_blob;
    }

   private:
    /// Returns dimension vector
    inline dims get_internal_dims() const {
      dim_t *c_dims = nullptr;
      dnnl_memory_desc_query(get(), dnnl_query_dims, &c_dims);
      return dims(c_dims, c_dims + get_internal_ndims());
    }

    // internal ndims
    inline int get_internal_ndims() const {
      return memory::desc::get_ndims();
    }

    bool is_blocking_desc() const {
      return get_format_kind() == format_kind::blocked;
    }

    bool is_opaque_desc() const {
      return get_format_kind() == format_kind::opaque;
    }

    void set_g(dim groups) {
      this->groups = groups;
    }

    dim g() const {
      return groups;
    }

    inline bool is_grouped() const {
      return g() > 1;
    }

    // We add the groups information into the last sizeof(int) item in the oneDNN blob vector
    // during the serialization.
    // We exclude the groups blob here when constructing oneDNN md.
    std::vector<uint8_t> get_dnnl_blob(const std::vector<uint8_t> &blob) {
      std::vector<uint8_t> dnnl_blob(blob.begin(), blob.end() - sizeof(int));
      return dnnl_blob;
    }

    int groups = 1;
  };

  desc get_desc() const {
    auto ret = desc(memory::get_desc());
    ret.set_g(groups_);
    return ret;
  }

  // Constructs an tensor with no buffer and zero memory description
  ITensor() {
    init({}, nullptr);
  }

  /// Constructs a tensor.
  ///
  /// @param desc tensor descriptor.
  /// @param aengine Engine.
  /// @param ahandle handle.
  ITensor(
      const desc& adesc,
      void* ahandle,
      const Engine& aengine = Engine::cpu_engine()) {
    init(adesc, ahandle, aengine);
  }

  /// Constructs a memory.
  ///
  /// @param desc tensor descriptor.
  /// @param aengine Engine.
  ITensor(const desc& adesc, const Engine& aengine = Engine::cpu_engine()) {
    init(adesc, aengine);
  }

  // format_tag, buffer
  ITensor(
      const dims& adims,
      data_type adata_type,
      format_tag aformat_tag,
      void* ahandle,
      const Engine& aengine = Engine::cpu_engine()) {
    init(adims, adata_type, aformat_tag, ahandle, aengine);
  }

  // format_tag, no buffer
  ITensor(
      const dims& adims,
      data_type adata_type,
      format_tag aformat_tag,
      const Engine& aengine = Engine::cpu_engine()) {
    init(adims, adata_type, aformat_tag, aengine);
  }

  // no format_tag, buffer
  ITensor(
      const dims& adims,
      data_type adata_type,
      void* ahandle,
      const Engine& aengine = Engine::cpu_engine()) {
    init(adims, adata_type, ahandle, aengine);
  }

  // no format_tag, no buffer
  ITensor(
      const dims& adims,
      data_type adata_type,
      const Engine& aengine = Engine::cpu_engine()) {
    init(adims, adata_type, aengine);
  }

  ITensor(
      const std::vector<float>& scales,
      const Engine& aengine = Engine::cpu_engine()) {
    init({(int)scales.size()}, data_type::f32, aengine);
    auto data_ptr = reinterpret_cast<float *>(get_data_handle());
    for (size_t i = 0; i < scales.size(); ++i) // fill in zero point data
      data_ptr[i] = scales[i];
  }

  ITensor(
      const std::vector<int32_t>& zero_points,
      const Engine& aengine = Engine::cpu_engine()) {
    init({(int)zero_points.size()}, data_type::s32, aengine);
    auto data_ptr = reinterpret_cast<int32_t *>(get_data_handle());
    for (size_t i = 0; i < zero_points.size(); ++i) // fill in zero point data
      data_ptr[i] = zero_points[i];
  }

  /// Function that refill tensor with new description. Specify extra buffer.
  void init(
      const desc& adesc,
      void* ahandle,
      const Engine& aengine = Engine::cpu_engine()) {
    buffer_.reset();
    scale_.reset();
    zero_point_.reset();
    eng_ = aengine;
    groups_ = adesc.g();
    reset_internal(adesc, aengine, ahandle);
  }

  /// Function that refill tensor with new description or buffer
  void init(const desc& adesc, const Engine& aengine = Engine::cpu_engine()) {
    buffer_.reset(aengine.malloc(adesc.get_size()), aengine.free);
    scale_.reset();
    zero_point_.reset();
    eng_ = aengine;
    groups_ = adesc.g();
    reset_internal(adesc, aengine, buffer_.get());
  }

  void zero_init(
      const desc& adesc,
      const Engine& aengine = Engine::cpu_engine()) {
    void* data = aengine.malloc(adesc.get_size());
    memset(data, 0, adesc.get_size());
    buffer_.reset(data, aengine.free);
    scale_.reset();
    zero_point_.reset();
    eng_ = aengine;
    groups_ = adesc.g();
    reset_internal(adesc, aengine, buffer_.get());
  }

  // format_tag, buffer
  void init(
      const dims& adims,
      data_type adata_type,
      format_tag aformat_tag,
      void* ahandle,
      const Engine& aengine = Engine::cpu_engine()) {
    init({adims, adata_type, aformat_tag}, ahandle, aengine);
  }

  // format_tag, no buffer
  void init(
      const dims& adims,
      data_type adata_type,
      format_tag aformat_tag,
      const Engine& aengine = Engine::cpu_engine()) {
    init({adims, adata_type, aformat_tag}, aengine);
  }

  // no format_tag, buffer
  void init(
      const dims& adims,
      data_type adata_type,
      void* ahandle,
      const Engine& aengine = Engine::cpu_engine()) {
    init({adims, adata_type, get_default_format(adims)}, ahandle, aengine);
  }

  // no format_tag, no buffer
  void init(
      const dims& adims,
      data_type adata_type,
      const Engine& aengine = Engine::cpu_engine()) {
    init({adims, adata_type, get_default_format(adims)}, aengine);
  }

  void reinit_if_possible(const desc& expected_desc) {
    auto curr_desc = get_desc();
    if (expected_desc != curr_desc) {
      if (curr_desc.has_same_shape_as(expected_desc)) {
        to_format(expected_desc);
      } else {
        init(expected_desc, get_engine());
      }
    }
  }

  /// Copy constructor
  ITensor(const ITensor& t)
      : memory(t),
        workspace_(t.workspace_),
        scale_(t.scale_),
        zero_point_(t.zero_point_),
        buffer_(t.buffer_),
        eng_(t.eng_),
        groups_(t.groups_) {}

  /// Move constructor
  ITensor(ITensor&& t)
      : memory(std::move(t)),
        workspace_(std::move(t.workspace_)),
        scale_(std::move(t.scale_)),
        zero_point_(std::move(t.zero_point_)),
        buffer_(std::move(t.buffer_)),
        eng_(std::move(t.eng_)),
        groups_(t.groups_) {}

  /// Assignment operator
  ITensor& operator=(const ITensor& t) {
    memory::operator=(t);
    buffer_ = t.buffer_;
    scale_ = t.scale_;
    zero_point_ = t.zero_point_;
    workspace_ = t.workspace_;
    eng_ = t.eng_;
    groups_ = t.groups_;
    return *this;
  }

  /// Move assignment operator
  ITensor& operator=(ITensor&& t) {
    memory::operator=(std::move(t));
    buffer_ = std::move(t.buffer_);
    scale_ = std::move(t.scale_);
    zero_point_ = std::move(t.zero_point_);
    workspace_ = std::move(t.workspace_);
    eng_ = std::move(t.eng_);
    groups_ = std::move(t.groups_);
    return *this;
  }

  /// Returns the engine of the tensor
  const Engine& get_engine() const {
    return eng_;
  }

  /// Returns number of dimensions
  inline int ndims() const {
    return get_desc().get_ndims();
  }

  /// Return size of specified dimension
  inline dim_t get_dim(int index) const {
    return get_desc().get_dim(index);
  }

  /// Returns dimension vector
  inline dims get_dims() const {
    return get_desc().get_dims();
  }

  inline dims get_strides() const {
    return get_desc().get_strides();
  }

  /// Return element number of the param.
  /// The number is the meaning values for a tensor, instead of whole buffer.
  /// It is the number without counting in paddings.
  inline dim_t get_nelems() const {
    return get_desc().nelems();
  }

  /// Returns descriptor data type
  inline data_type get_data_type() const {
    return get_desc().get_data_type();
  }

  inline size_t get_size() const {
    return get_desc().get_size();
  }

  /// Return whether the tensor is empty
  inline bool is_empty() const {
    return get_desc().is_zero() && get_data_handle() == nullptr;
  }

  // "public format" has the same semantic as DNNL's "plain format"
  inline bool is_public_format() const {
    return get_desc().is_plain();
  }

  inline static format_tag get_default_format(const dims& adims) {
    switch (adims.size()) {
      case 1:
        return format_tag::a;
      case 2:
        return format_tag::ab;
      case 3:
        return format_tag::abc;
      case 4:
        return format_tag::abcd;
      case 5:
        return format_tag::abcde;
      case 6:
        return format_tag::abcdef;
      default:
        return format_tag::undef;
    }
  }

  ITensor reorder_if_differ_in(
      const desc& expected_desc,
      const Attr& aattr = Attr()) const {
    // Check desc, scales and zero points
    bool reorder_not_needed = (expected_desc == get_desc());
    if (reorder_not_needed && aattr.has_scales()) {
      for (auto& arg_scale_pair : aattr.get_all_scales()) {
        const std::vector<float>& scales = arg_scale_pair.second.first;
        reorder_not_needed = reorder_not_needed &&
            (scales.empty() ||
             std::all_of(scales.begin(), scales.end(), [](float i) {return 1.0 == i;}));
      }
    }
    if (reorder_not_needed && aattr.has_zero_points()) {
      for (auto& arg_zp_pair : aattr.get_all_zero_points()) {
        const std::vector<int32_t>& zp = arg_zp_pair.second.first;
        reorder_not_needed = reorder_not_needed &&
            (zp.empty() ||
             std::all_of(zp.begin(), zp.end(), [](int i) {return 0 == i;}));
      }
    }
    if (reorder_not_needed) {
      return *this;
    } else {
      ITensor dst{expected_desc};
      // Keep scale and zero point
      if (has_scale()) {
        dst.set_scale(get_scale());
      }
      if (has_zero_point()) {
        dst.set_zero_point(get_zero_point());
      }
      // Try to reorder and catch possible runtime errors.
      // If error occurs, it is reordered to plain format then to the desired
      // format
      try {
        reorder_to(dst, aattr);
      } catch (...) {
        // A common error is 'could not create a reorder primitive descriptor'
        // We won't distinguish between specific errors
        ITensor&& plain_weight = to_public(nullptr, get_data_type());
        plain_weight.reorder_to(dst, aattr);
      }
      return dst;
    }
  }

  // workaround for issue intel/mkl-dnn#588
  desc _get_unblocked_desc_if_4c_blocked() const {
    auto desc = get_desc();
    return desc.is_4c_blocked() ? desc.to_default_format() : desc;
  }

  // no data copy
  ITensor make_grouped_weights(int groups, bool is_deconv = false) const {
    if (groups <= 1)
      return *this;

    auto old_desc = get_desc();
    auto old_groups = old_desc.g();
    if (old_groups > 1) {
      // weights have already been pre-converted if old_groups > 1
#ifndef NDEBUG
      dnnl::error::wrap_c_api(
          (old_groups == groups) ? dnnl_success : dnnl_invalid_arguments,
          "groups does not match the pre-converted weights");
#endif
      return *this;
    }

    auto grouped_desc = is_deconv
        ? old_desc.transpose(0, 1).to_grouped(groups).transpose(1, 2)
        : old_desc.to_grouped(groups);

    // handle channels last with groups
    if (is_deconv) {
      // deconv: judge whether is channels last on iohw format
      auto old_desc_trans = old_desc.transpose(0, 1);
      if (old_desc_trans.is_nhwc()) {
        // giohw (acbde) => gihwo (acdeb)
        grouped_desc = grouped_desc.to_format(format_tag::acdeb);
        grouped_desc.set_g(groups);
      } else if (old_desc_trans.is_ndhwc()) {
        // giodhw (acbdef) => gidhwo (acdefb)
        // TODO: onednn doesn't have the tag of acdefb for now
        // grouped_desc = grouped_desc.to_format(format_tag::acdefb);
        //
        // work around by re-create desc based on dims and strides.
        auto ddims = grouped_desc.get_dims();
        auto ddata_type = grouped_desc.get_data_type();
        auto g = groups;
        auto o = ddims[0] / g;
        auto i = ddims[1];
        auto d = ddims[2];
        auto h = ddims[3];
        auto w = ddims[4];
        desc new_desc{
            {g, o, i, d, h, w},
            ddata_type,
            {/*g*/ i * d * h * w * o,
             /*o*/ 1,
             /*i*/ d * h * w * o,
             /*d*/ h * w * o,
             /*h*/ w * o,
             /*w*/ o}};
        grouped_desc = new_desc;
        grouped_desc.set_g(groups);
      }
    } else {
      // conv: judge whether is channels last on oihw format
      auto channels_last = old_desc.is_channels_last();
      if (channels_last) {
        // goihw (abcde) => gohwi (abdec) or goidhw (abcdef) => gohwi (abdefc)
        auto memory_format = format_tag::abdec;
        auto dim = old_desc.get_ndims();
        if (dim == 5) {
          memory_format = format_tag::abdefc;
        } else if (dim == 3) {
          memory_format = format_tag::abdc;
        }
        grouped_desc = grouped_desc.to_format(memory_format);
      }
    }

    auto this_copy = *this;
    return this_copy.set_desc(grouped_desc);
  }

  /// Return an new tensor with new shape
  ITensor& reshape(const dims& adims) {
#ifndef NDEBUG
    dnnl::error::wrap_c_api(
        (has_same_volume(adims)) ? dnnl_success : dnnl_invalid_arguments,
        "reshape to incompatible shape");
#endif

    auto need_convert_to_default_format = [](const desc& src_desc,
                                             const dims& shape) {
      // if src_desc is default format, do not need to convert format.
      if (src_desc.is_default()) {
        return false;
      } else {
        // count the number of non-one dimensions
        // e.g. the squeezed_ndims of shape [1, 1, 35, 1] is one.
        auto squeezed_ndims = 0;
        for (auto d : shape)
          if (d > 1)
            squeezed_ndims++;
        if (squeezed_ndims == 0)
          return false; // [1, 1, ...]
        // For squeezed_ndims is one, src_desc is plain format
        // or src_desc is block format, but the blocking dim's size is not one,
        // for example, aBcd16b, the shape is [1, 2048, 1, 1], the blocking dim
        // is the second dimension, the strid is one for the blockind dim, the
        // format does not matter for data idexing. But for aBc16b with shape
        // [1, 1, 7], we need do format change even the squeezed_ndims is one,
        // because the last dimension is not contiguous, the stride is 16.
        if (squeezed_ndims == 1) {
          if (src_desc.is_plain())
            return false;
          // block format, only one dim is blocked, and the size of blocked dim
          // > 1.
          // auto block_desc = src_desc.blocking_desc();
          if (src_desc.get_inner_nblks() == 1 &&
              shape[src_desc.get_inner_idxs()[0]] > 1) {
            return false;
          }
        }
        return true;
      }
    };
    auto old_dims = get_dims();
    if (adims != old_dims) {
      if (need_convert_to_default_format(get_desc(), old_dims)) {
        to_default_format();
      }
      // set desc with default format
      set_desc({adims, get_data_type()});
    }
    return *this;
  }

  inline void to_default_format() {
    to_format(get_desc().to_default_format());
  }

  inline void to_format(format_tag aformat_tag) {
    to_format(get_desc().to_format(aformat_tag));
  }

  // TODO(xpz): not a good name
  inline void to_type(data_type adata_type) {
    set_desc(get_desc().to_type(adata_type));
  }

  inline void reorder_to(ITensor& dst, const Attr& aattr = Attr()) const {
    Attr op_attr = aattr;
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pd = dnnl::reorder::primitive_desc(*this, dst, op_attr);

    ITensor scratchpad(pd.scratchpad_desc());
    std::unordered_map<int, dnnl::memory> args;
    args.insert({DNNL_ARG_FROM, const_cast<ITensor&>(*this)});
    args.insert({DNNL_ARG_TO, dst});
    args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
    // Insert scales and zero points
    const static std::vector<int> args_for_scales_zp = {DNNL_ARG_SRC, DNNL_ARG_DST};
    std::vector<ITensor> scale_tensors(args_for_scales_zp.size());
    for (size_t i = 0; i < args_for_scales_zp.size(); ++i) {
      int arg = args_for_scales_zp[i];
      if (op_attr.has_scales_for(arg)) {
        scale_tensors[i] = ITensor(op_attr.get_scales(arg).first);
        args.insert({DNNL_ARG_ATTR_SCALES | arg, scale_tensors[i]});
      }
    }
    std::vector<ITensor> zp_tensors(args_for_scales_zp.size());
    for (size_t i = 0; i < args_for_scales_zp.size(); ++i) {
      int arg = args_for_scales_zp[i];
      if (op_attr.has_zero_points_for(arg)) {
        zp_tensors[i] = ITensor(op_attr.get_zero_points(arg).first);
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | arg, zp_tensors[i]});
      }
    }
    dnnl::reorder(pd).execute(
        IStream::default_stream(),
        args);
  }

  /// Convert the tensor to public format, and f32 data type by default
  ITensor to_public(void* buffer = nullptr, data_type dst_type = data_type::f32)
      const {
    auto dst_desc = get_desc();

    // If we get a non-plain blocking format, say `Acdb16A`, we may not be able
    // to recover it to its "unblocked" format `acdb`. Instead, we will convert
    // it to its default format `abcd` based on its dimensions.
    if (!is_public_format()) {
      dst_desc = dst_desc.to_default_format();
    }

    if (dst_type != data_type::undef) {
      dst_desc = dst_desc.to_type(dst_type);
    }

    auto dst = buffer ? ITensor(dst_desc, buffer) : ITensor(dst_desc);

    if (one_of(
            get_data_type(), data_type::s8, data_type::u8, data_type::s32) &&
        dst_desc.get_data_type() == data_type::f32 && has_scale()) {
      auto& src_scale = get_scale();
      auto mask =
          utils::tensor_scale_mask(src_scale.size(), get_desc().is_grouped());
      this->reorder_to(dst, {mask, src_scale});
    } else {
      this->reorder_to(dst);
      if (has_scale()) {
        dst.set_scale(get_scale());
      }
    }

    return dst;
  }

  /// Fill the tensor with a src tensor
  /// TODO(xpz): may replace is_deconv_weights with a enum for other purposes
  void feed_from(const ITensor& src, bool is_deconv_weights = false) {
    std::vector<float> dst_scale, src_scale;
    if (has_scale() && src.has_scale()) {
      dst_scale = get_scale();
      src_scale = src.get_scale();
    } else if (has_scale()) {
      dst_scale = get_scale();
      src_scale.assign(dst_scale.size(), 1.0f);
    } else if (src.has_scale()) {
      src_scale = src.get_scale();
      dst_scale.assign(src_scale.size(), 1.0f);
    }
#ifndef NDEBUG
    dnnl::error::wrap_c_api(
        (dst_scale.size() == src_scale.size()) ? dnnl_success : dnnl_invalid_arguments,
        "Invalid tensor scales");
#endif
    std::vector<float> scales(dst_scale.size());
    for (size_t i = 0; i < dst_scale.size(); i++) {
      scales[i] = src_scale[i] / dst_scale[i];
    }

    auto groups = 1;
    if ((groups = get_desc().g()) > 1 || (groups = src.get_desc().g()) > 1) {
      auto mask_dst = this->make_grouped_weights(groups, is_deconv_weights);
      auto mask_src = src.make_grouped_weights(groups, is_deconv_weights);
      int mask = utils::tensor_scale_mask(src_scale.size(), true);
      if (scales.empty()) {
        mask_src.reorder_to(mask_dst);
      } else {
        mask_src.reorder_to(mask_dst, {mask, scales});
      }
    } else {
      int mask = utils::tensor_scale_mask(src_scale.size(), false);
      if (scales.empty()) {
        src.reorder_to(*this);
      } else {
        src.reorder_to(*this, {mask, scales});
      }
    }
  }

  // For backward compatibility. Will be deprecated.
  void feed_from(const dims& adims, data_type adata_type, const void* array) {
    feed_from({adims, adata_type, const_cast<void*>(array)});
  }

  ITensor dequantize() const {
    ITensor dst(get_desc().to_type(data_type::f32));
#ifndef NDEBUG
    dnnl::error::wrap_c_api(
        (has_scale()) ? dnnl_success : dnnl_invalid_arguments,
        "Can not find scales");
#endif
    // TODO(xpz): support per-channel dequantize
#ifndef NDEBUG
    dnnl::error::wrap_c_api(
        (get_scale().size() == 1) ? dnnl_success : dnnl_invalid_arguments,
        "Incorrect scale size");
#endif
    dst.feed_from(*this);
    return dst;
  }

  void init_workspace(const desc& desc) {
    auto workspace = new ITensor(desc, get_engine());
    workspace_.reset(workspace);
  }

  /// Return extra packed tensor
  ITensor& get_workspace() const {
    return *workspace_;
  }

  /// Decide whether there is an extra tensor packed in
  bool has_workspace() const {
    return workspace_ != nullptr;
  }

  /// Return the scale of this param.
  const std::vector<float>& get_scale() const {
    return *scale_.get();
  }

  /// Set new scale into param
  void set_scale(const std::vector<float>& ascale) {
    scale_.reset(new std::vector<float>(ascale));
  }

  /// Return whether the param has a scale
  bool has_scale() const {
    return scale_ != nullptr && !scale_->empty();
  }

  /// Return whether the param has a zero_point
  bool has_zero_point() const {
    return zero_point_ != nullptr && !zero_point_->empty();
  }

  /// Return the zero_point of this param.
  const std::vector<int32_t>& get_zero_point() const {
    return *zero_point_.get();
  }

  /// Set new scale into param
  void set_zero_point(const std::vector<int32_t>& zp) {
    zero_point_.reset(new std::vector<int32_t>(zp));
  }

  ITensor& permute_(const std::vector<int>& permute_axes = {}) {
    return set_desc(get_desc().permute(permute_axes));
  }

  ITensor permute(const std::vector<int>& permute_axes = {}) const {
    auto src_mask = *this;
    src_mask.permute_(permute_axes);
    auto dst = ITensor(src_mask.get_desc().to_default_format());
    src_mask.reorder_to(dst);
    return dst;
  }

  ITensor& transpose_(dim dim0, dim dim1) {
    return set_desc(get_desc().transpose(dim0, dim1));
  }

  ITensor transpose(dim dim0, dim dim1) const {
    auto src_mask = *this;
    src_mask.transpose_(dim0, dim1);
    auto dst = ITensor(src_mask.get_desc().to_default_format());
    src_mask.reorder_to(dst);
    return dst;
  }

  // For backward compatibility. Will be deprecated
  void transpose_from(const ITensor& src, const std::vector<int>& perms = {}) {
    *this = std::move(src.permute(perms));
  }

 private:
  void reset_internal(const desc& adesc, const Engine& aengine, void* ahandle) {
    dnnl_memory_t result;
    dnnl_memory_desc_t memory_desc = adesc.get();
    dnnl::error::wrap_c_api(
        dnnl_memory_create(&result, memory_desc, aengine.get(), ahandle),
        "could not create a memory");
    reset(result);
  }

  inline void to_format(const desc& adesc) {
    if (get_desc() != adesc) {
      auto dst = ITensor(adesc);
      this->reorder_to(dst);
      *this = std::move(dst);
    }
  }

  bool has_same_volume(const dims& new_dims) const {
    auto old_dims = get_dims();
    auto volume_old = std::accumulate(
        old_dims.begin(), old_dims.end(), 1, std::multiplies<dim_t>());
    auto volume_new = std::accumulate(
        new_dims.begin(), new_dims.end(), 1, std::multiplies<dim_t>());
    return volume_old == volume_new;
  }

  /// Set a descriptor into tensor to replace the older one, keep buffer
  /// It is caller's responsibility to make sure the original buffer is large
  /// enough for specified descriptor
  ITensor& set_desc(const desc& new_desc) {
    // Keep the original management
    auto buf = std::move(buffer_);
    auto ws = std::move(workspace_);
    auto scale = std::move(scale_);
    auto zp = std::move(zero_point_);
    init(new_desc, get_data_handle(), get_engine());
    buffer_ = std::move(buf);
    workspace_ = std::move(ws);
    scale_ = std::move(scale);
    zero_point_ = std::move(zp);
    return *this;
  }

  std::shared_ptr<ITensor> workspace_;
  std::shared_ptr<std::vector<float>> scale_;
  std::shared_ptr<std::vector<int32_t>> zero_point_;
  std::shared_ptr<void> buffer_;
  Engine eng_;
  int groups_;
};

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
