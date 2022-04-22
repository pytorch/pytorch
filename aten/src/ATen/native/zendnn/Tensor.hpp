#pragma once

#include "ATen/native/zendnn/Attributes.hpp"
#include "ATen/native/zendnn/Utils.hpp"

namespace zendnn {

class tensor : public memory {
 public:
  using dim_t = zendnn_dim_t;
  using dims_t = zendnn_dims_t;
  using format_kind_t = zendnn_format_kind_t;
  using blocking_desc_t = zendnn_blocking_desc_t;
  using descriptor = tensor::desc; // for backward compatibility

  struct desc : public memory::desc {
    friend class tensor;

    // avoid conflicts with function desc::dims() and desc::data_type()
    using dims = typename memory::dims;
    using data_type = typename memory::data_type;

    desc() : memory::desc(){};

    // copy ctor
    desc(const desc& adesc) : memory::desc(adesc.data) {
      set_g(adesc.g());
    };

    // supplement group info for memory::desc
    desc(const memory::desc& adesc, dim groups = 1) : memory::desc(adesc.data) {
      set_g(groups);
    };

    desc& operator=(const desc& adesc) {
      memory::desc::operator=(adesc);
      set_g(adesc.g());
      return *this;
    }

    desc(const zendnn_memory_desc_t& adata) : memory::desc(adata){};

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

    void to_bytes(utils::bytestring& bytes) const {
      utils::to_bytes(bytes, data_type());
      utils::to_bytes(bytes, format_kind());
      utils::to_bytes(bytes, offset0());

      auto& paddim = padded_dims();
      auto& padoff = padded_offsets();

      for (int i = 0; i < data.ndims; i++) {
        utils::to_bytes(bytes, data.dims[i]);
        utils::to_bytes(bytes, paddim[i]);
        utils::to_bytes(bytes, padoff[i]);
      }

      if (is_blocking_desc()) {
        auto& blk = blocking_desc();
        auto& stride = blocking_strides();
        for (int i = 0; i < data.ndims; i++) {
          utils::to_bytes(bytes, stride[i]);
        }
        for (int i = 0; i < blk.inner_nblks; i++) {
          utils::to_bytes(bytes, blk.inner_idxs[i]);
          utils::to_bytes(bytes, blk.inner_blks[i]);
        }
      }
    }

    /// public ndims
    inline int get_ndims() const {
      return is_grouped() ? data.ndims - 1 : data.ndims;
    }

    /// Return size of specified dimension
    inline dim_t get_dim(int index) const {
      if (!is_grouped()) {
        if (index < 0 || index >= data.ndims)
          return static_cast<dim_t>(0);
        return data.dims[index];
      } else {
        if (index < 0 || index >= data.ndims - 1)
          return static_cast<dim_t>(0);
        return index == 0 ? data.dims[0] * data.dims[1] : data.dims[index + 1];
      }
    }

    /// Returns dimension vector
    inline dims get_dims() const {
      if (!is_grouped()) {
        return dims(data.dims, data.dims + data.ndims);
      } else {
        auto ret = dims(data.dims + 1, data.dims + data.ndims);
        ret[0] *= data.dims[0]; // g == data.dims[0]
        return ret;
      }
    }

    /// Returns descriptor data type
    inline data_type get_data_type() const {
      return static_cast<data_type>(data.data_type);
    }

    inline dims get_strides() const {
      ZENDNN_ENFORCE(is_plain(), "Call to_public() before get_strides()");
      const auto& strides = blocking_strides();
      if (!is_grouped()) {
        return dims(strides, strides + data.ndims);
      } else {
        auto ret = dims(strides + 1, strides + data.ndims);
        ret[0] = std::min(strides[0], strides[1]);
        return ret;
      }
    }

    /** returns true if memory descriptor is zero */
    bool is_zero() const {
      return data.ndims == 0;
    }

    /** returns the number of elements including padding if \param with_padding
     * is true, and the number of data elements otherwise */
    inline dim_t nelems(bool with_padding = false) const {
      if (is_zero())
        return 0;
      auto dims = with_padding ? data.padded_dims : data.dims;
      return std::accumulate(
          dims, dims + data.ndims, 1, std::multiplies<dim_t>());
    }

    inline bool is_plain() const {
      return is_blocking_desc() && blocking_desc().inner_nblks == 0;
    };

    inline bool is_default() const {
      if (!is_plain())
        return false;

      const auto& strides = blocking_strides();
      for (int i = 0; i < data.ndims - 1; i++) {
        if (strides[i] < strides[i + 1]) {
          return false;
        }
      }
      return true;
    }

    inline bool is_nhwc() const {
      if (!is_plain() || data.ndims != 4)
        return false;
      const auto& dims = data.dims;
      const auto& strides = blocking_strides();
      const auto n = 0, c = 1, h = 2, w = 3;
      return strides[n] == dims[h] * dims[w] * dims[c] &&
          strides[h] == dims[w] * dims[c] && strides[w] == dims[c] &&
          strides[c] == 1;
    };

    inline bool is_nchw() const {
      if (!is_plain() || data.ndims != 4)
        return false;
      const auto& dims = data.dims;
      const auto& strides = blocking_strides();
      const auto n = 0, c = 1, h = 2, w = 3;
      return strides[n] == dims[c] * dims[h] * dims[w] &&
          strides[c] == dims[h] * dims[w] && strides[h] == dims[w] &&
          strides[w] == 1;
    };

    inline bool is_iohw() const {
      if (!is_plain() || data.ndims != 4)
        return false;
      const auto& dims = data.dims;
      const auto& strides = blocking_strides();
      const auto o = 0, i = 1, h = 2, w = 3;
      return strides[i] == dims[o] * dims[h] * dims[w] &&
          strides[o] == dims[h] * dims[w] && strides[h] == dims[w] &&
          strides[w] == 1;
    };

    // workaround for issue intel/mkl-dnn#588
    bool is_4c_blocked() {
      const auto& blk = blocking_desc();
      return blk.inner_nblks == 1 && blk.inner_idxs[0] == 1 &&
          blk.inner_blks[0] == 4;
    }

    // legacy API for caffe2
    bool is_limited_blockable() const {
      const auto& blk = blocking_desc();
      // compute compatible block_dims with v0.x
      dims block_dims(data.ndims, 1);
      for (auto i = 0; i < blk.inner_nblks; i++) {
        block_dims[blk.inner_idxs[i]] *= blk.inner_blks[i];
      }
      for (auto i = 0; i < data.ndims; i++) {
        if (data.dims[i] < block_dims[i])
          continue;
        if (data.dims[i] % block_dims[i] == 0)
          continue;
        return false;
      }
      return true;
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

    desc to_type(data_type atype) const {
      auto ret = clone();
      ret.data.data_type = static_cast<zendnn_data_type_t>(atype);
      ret.set_g(g());
      return ret;
    }

    desc to_grouped(int groups, bool is_deconv = false) const {
      auto grouped_dims = utils::group_dims(get_internal_dims(), groups);
      auto grouped_desc = desc(grouped_dims, get_data_type());
      grouped_desc.set_g(groups);
      return grouped_desc;
    }

    bool has_same_shape_as(const desc& that) const {
      if (data.ndims != that.data.ndims)
        return false;
      return utils::array_cmp(data.dims, that.data.dims, data.ndims);
    }

    // to be replaced with memory_desc_permute_axes in ZENDNN v1.3
    desc permute(const std::vector<int>& permute_axes = {}) const {
      if (data.ndims <= 1) {
        return clone();
      }

      auto perms = permute_axes;
      if (perms.empty()) {
        perms.resize(data.ndims);
        std::iota(perms.rbegin(), perms.rend(), 0);
      } else {
        ZENDNN_ENFORCE(
            perms.size() == data.ndims,
            "Axes should be size like source tensor.");
        auto perms_sorted = perms;
        std::sort(perms_sorted.begin(), perms_sorted.end());
        for (auto i = 0; i < perms_sorted.size(); ++i) {
          ZENDNN_ENFORCE(
              perms_sorted[i] == i,
              "Axes should be a permutation of 0 to ndim.");
        }
        if (perms_sorted == perms) {
          return clone();
        }
      }

      desc new_desc{};
      auto ndims = data.ndims;
      new_desc.data.ndims = data.ndims;
      new_desc.data.data_type = data.data_type;
      new_desc.data.format_kind = data.format_kind;
      new_desc.data.offset0 = data.offset0;
      new_desc.set_g(g());

      // permute dims, padded_dims, padded_offsets, strides
      auto& new_dims = new_desc.data.dims;
      auto& old_dims = data.dims;
      auto& new_stride = new_desc.data.format_desc.blocking.strides;
      auto& old_stride = data.format_desc.blocking.strides;
      auto& new_paddim = new_desc.data.padded_dims;
      auto& old_paddim = data.padded_dims;
      auto& new_padoff = new_desc.data.padded_offsets;
      auto& old_padoff = data.padded_offsets;
      for (int i = 0; i < ndims; i++) {
        new_dims[i] = old_dims[perms[i]];
        new_stride[i] = old_stride[perms[i]];
        new_paddim[i] = old_paddim[perms[i]];
        new_padoff[i] = old_padoff[perms[i]];
      }

      // permute blocking
      auto inner_nblks = data.format_desc.blocking.inner_nblks;
      new_desc.data.format_desc.blocking.inner_nblks = inner_nblks;
      auto& old_inner_idxs = data.format_desc.blocking.inner_idxs;
      auto& new_inner_idxs = new_desc.data.format_desc.blocking.inner_idxs;
      auto& old_inner_blks = data.format_desc.blocking.inner_blks;
      auto& new_inner_blks = new_desc.data.format_desc.blocking.inner_blks;
      for (int i = 0; i < inner_nblks; i++) {
        new_inner_idxs[i] = perms[old_inner_idxs[i]];
        new_inner_blks[i] = old_inner_blks[i];
      }

      return new_desc;
    }

    desc transpose(dim dim0, dim dim1) const {
      std::vector<int> axes(data.ndims);
      std::iota(axes.begin(), axes.end(), 0);
      std::swap(axes[dim0], axes[dim1]);
      return permute(axes);
    }

    /** inits descriptor with logical dimensions adims and keep the current
     * blocking structure
     */
    desc to_dims(const dims& adims) const {
      ZENDNN_ENFORCE(adims.size() == data.ndims, "Rank mismatched.");

      zendnn_memory_desc_t md;
      md.ndims = data.ndims;
      md.data_type = data.data_type;

      auto& blk = blocking_desc();

      dims_t blocks;
      for (auto i = 0; i < data.ndims; i++)
        blocks[i] = 1;

      dim_t block_size = 1;
      for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
        blocks[blk.inner_idxs[iblk]] *= blk.inner_blks[iblk];
        block_size *= blk.inner_blks[iblk];
      }

      for (int d = 0; d < data.ndims; ++d) {
        md.dims[d] = adims[d];
        md.padded_dims[d] = utils::rnd_up(adims[d], blocks[d]);
        md.padded_offsets[d] = 0;
      }
      md.offset0 = 0;

      md.format_kind = zendnn_blocked;
      auto& mblk = md.format_desc.blocking;
      mblk = blk;

      for (auto i = 0; i < data.ndims; i++)
        mblk.strides[i] = blk.strides[i];

      int perm[ZENDNN_MAX_NDIMS];
      for (int d = 0; d < data.ndims; ++d)
        perm[d] = d;

      utils::simultaneous_sort(
          mblk.strides, perm, data.ndims, [](dim_t a, dim_t b) {
            return b - a;
          });

      dim_t stride = block_size;
      for (int _d = data.ndims - 1; _d >= 0; --_d) {
        const int d = perm[_d];
        md.format_desc.blocking.strides[d] = stride;
        stride *= md.padded_dims[d] / blocks[d];
      }

      md.extra = zendnn_memory_extra_desc_t{};

      return desc(md);
    }

   private:
    /// Returns dimension vector
    inline dims get_internal_dims() const {
      return dims(data.dims, data.dims + data.ndims);
    }

    const dims_t& padded_dims() const {
      return data.padded_dims;
    }

    const dims_t& padded_offsets() const {
      return data.padded_offsets;
    }

    dim_t offset0() const {
      return data.offset0;
    }

    inline format_kind_t format_kind() const {
      return data.format_kind;
    }

    bool is_blocking_desc() const {
      return format_kind() == zendnn_blocked;
    }

    bool is_wino_desc() const {
      return format_kind() == zendnn_format_kind_wino;
    }

    bool is_rnn_packed_desc() const {
      return format_kind() == zendnn_format_kind_rnn_packed;
    }

    const blocking_desc_t& blocking_desc() const {
      ZENDNN_ENFORCE(
          is_blocking_desc(),
          "Cannot get blocking desc on a non-blocking desc");
      return data.format_desc.blocking;
    }

    dims_t& blocking_strides() const {
      ZENDNN_ENFORCE(
          is_blocking_desc(),
          "Cannot get blocking desc on a non-blocking desc");
      return const_cast<zendnn_memory_desc_t&>(data)
          .format_desc.blocking.strides;
    }

    void set_g(dim groups) {
      auto reserved_size = sizeof(((zendnn_memory_extra_desc_t*)0)->reserved);
      auto offset = reserved_size / sizeof(dim) - 1;
      reinterpret_cast<dim*>(data.extra.reserved)[offset] = groups;
    }

    dim g() const {
      auto reserved_size = sizeof(((zendnn_memory_extra_desc_t*)0)->reserved);
      auto offset = reserved_size / sizeof(dim) - 1;
      return reinterpret_cast<const dim*>(data.extra.reserved)[offset];
    }

    inline bool is_grouped() const {
      return g() > 1;
    }
  };

  desc get_desc() const {
    const zendnn_memory_desc_t* cdesc;
    error::wrap_c_api(
        zendnn_memory_get_memory_desc(get(), &cdesc),
        "could not get memory descriptor from a memory");
    return desc(*cdesc);
  }

  // For backward compatibility. Will be deprecated.
  desc get_descriptor() const {
    return get_desc();
  }

  desc dup_desc() const {
    return get_desc().clone();
  }

  // For backward compatibility. Will be deprecated.
  desc dup_descriptor() const {
    return dup_desc();
  }

  // Constructs an tensor with no buffer and zero memory description
  tensor() {
    init({}, nullptr);
  }

  /// Constructs a tensor.
  ///
  /// @param desc tensor descriptor.
  /// @param aengine Engine.
  /// @param ahandle handle.
  tensor(
      const desc& adesc,
      void* ahandle,
      const utils::engine& aengine = utils::engine::cpu_engine()) {
    init(adesc, ahandle, aengine);
  }

  /// Constructs a memory.
  ///
  /// @param desc tensor descriptor.
  /// @param aengine Engine.
  tensor(
      const desc& adesc,
      const utils::engine& aengine = utils::engine::cpu_engine()) {
    init(adesc, aengine);
  }

  // format_tag, buffer
  tensor(
      const dims& adims,
      data_type adata_type,
      format_tag aformat_tag,
      void* ahandle,
      const utils::engine& aengine = utils::engine::cpu_engine()) {
    init(adims, adata_type, aformat_tag, ahandle, aengine);
  }

  // format_tag, no buffer
  tensor(
      const dims& adims,
      data_type adata_type,
      format_tag aformat_tag,
      const utils::engine& aengine = utils::engine::cpu_engine()) {
    init(adims, adata_type, aformat_tag, aengine);
  }

  // no format_tag, buffer
  tensor(
      const dims& adims,
      data_type adata_type,
      void* ahandle,
      const utils::engine& aengine = utils::engine::cpu_engine()) {
    init(adims, adata_type, ahandle, aengine);
  }

  // no format_tag, no buffer
  tensor(
      const dims& adims,
      data_type adata_type,
      const utils::engine& aengine = utils::engine::cpu_engine()) {
    init(adims, adata_type, aengine);
  }

  /// Function that refill tensor with new description. Specifiy extra buffer.
  void init(
      const desc& adesc,
      void* ahandle,
      const utils::engine& aengine = utils::engine::cpu_engine()) {
    buffer_.reset();
    scale_.reset();
    zero_point_.reset();
    eng_ = aengine;
    reset_internal(adesc, aengine, ahandle);
  }

  /// Function that refill tensor with new description or buffer
  void init(
      const desc& adesc,
      const utils::engine& aengine = utils::engine::cpu_engine()) {
    buffer_.reset(aengine.malloc(adesc.get_size()), aengine.free);
    scale_.reset();
    zero_point_.reset();
    eng_ = aengine;
    reset_internal(adesc, aengine, buffer_.get());
  }

  // format_tag, buffer
  void init(
      const dims& adims,
      data_type adata_type,
      format_tag aformat_tag,
      void* ahandle,
      const utils::engine& aengine = utils::engine::cpu_engine()) {
    init({adims, adata_type, aformat_tag}, ahandle, aengine);
  }

  // format_tag, no buffer
  void init(
      const dims& adims,
      data_type adata_type,
      format_tag aformat_tag,
      const utils::engine& aengine = utils::engine::cpu_engine()) {
    init({adims, adata_type, aformat_tag}, aengine);
  }

  // no format_tag, buffer
  void init(
      const dims& adims,
      data_type adata_type,
      void* ahandle,
      const utils::engine& aengine = utils::engine::cpu_engine()) {
    init({adims, adata_type, get_default_format(adims)}, ahandle, aengine);
  }

  // no format_tag, no buffer
  void init(
      const dims& adims,
      data_type adata_type,
      const utils::engine& aengine = utils::engine::cpu_engine()) {
    init({adims, adata_type, get_default_format(adims)}, aengine);
  }

  // legacy API for caffe2
  void reinit_like(const tensor& t) {
    init(t.get_desc(), t.get_engine());
  }

  // legacy API for caffe2
  void reinit_like(const tensor& t, void* ahandle) {
    init(t.get_desc(), ahandle, t.get_engine());
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
  tensor(const tensor& t)
      : memory(t),
        workspace_(t.workspace_),
        scale_(t.scale_),
        zero_point_(t.zero_point_),
        buffer_(t.buffer_),
        eng_(t.eng_) {}

  /// Move constructor
  tensor(tensor&& t)
      : memory(std::move(t)),
        workspace_(std::move(t.workspace_)),
        scale_(std::move(t.scale_)),
        zero_point_(std::move(t.zero_point_)),
        buffer_(std::move(t.buffer_)),
        eng_(std::move(t.eng_)) {}

  /// Assignment operator
  tensor& operator=(const tensor& t) {
    memory::operator=(t);
    buffer_ = t.buffer_;
    scale_ = t.scale_;
    zero_point_ = t.zero_point_;
    workspace_ = t.workspace_;
    eng_ = t.eng_;
    return *this;
  }

  /// Move assignment operator
  tensor& operator=(tensor&& t) {
    memory::operator=(std::move(t));
    buffer_ = std::move(t.buffer_);
    scale_ = std::move(t.scale_);
    zero_point_ = std::move(t.zero_point_);
    workspace_ = std::move(t.workspace_);
    eng_ = std::move(t.eng_);
    return *this;
  }

  /// Returns the engine of the tensor
  const utils::engine& get_engine() const {
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

  // "public format" has the same semantic as ZENDNN's "plain format"
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

  // legacy API for caffe2
  dims get_public_format_dims() const {
    auto nchw_dims = get_dims();
    if (get_desc().is_nhwc()) {
      dims nhwc_dims(ndims());
      nhwc_dims[0] = nchw_dims[0];
      nhwc_dims[1] = nchw_dims[2];
      nhwc_dims[2] = nchw_dims[3];
      nhwc_dims[3] = nchw_dims[1];
      return nhwc_dims;
    }
    return nchw_dims;
  }

  tensor reorder_if_differ_in(
      const desc& expected_desc,
      const attr_t& aattr = attr_t()) const {
    if (expected_desc == get_desc()) {
      return *this;
    } else {
      tensor dst{expected_desc};
      this->reorder_to(dst, aattr);
      return dst;
    }
  }

  // workaround for issue intel/mkl-dnn#588
  desc _get_unblocked_desc_if_4c_blocked() const {
    auto desc = get_desc();
    return desc.is_4c_blocked() ? desc.to_default_format() : desc;
  }

  // no data copy
  tensor make_grouped_weights(int groups, bool is_deconv = false) const {
    if (groups <= 1)
      return *this;

    auto old_desc = get_desc();
    auto old_groups = old_desc.g();
    if (old_groups > 1) {
      // weights have already been pre-converted if old_groups > 1
      ZENDNN_ENFORCE(
          old_groups == groups,
          "groups does not match the pre-converted weights");
      return *this;
    }

    auto grouped_desc = is_deconv
        ? old_desc.transpose(0, 1).to_grouped(groups).transpose(1, 2)
        : old_desc.to_grouped(groups);

    // handle channels last with groups
    if (is_deconv) {
      // deconv: judge whether is channels last on iohw format
      auto is_channels_last = old_desc.transpose(0, 1).is_nhwc();
      if (is_channels_last) {
        // giohw (acbde) => gihwo (acdeb)
        grouped_desc = grouped_desc.to_format(format_tag::acdeb);
      }
    } else {
      // conv: judge whether is channels last on oihw format
      auto is_channels_last = old_desc.is_nhwc();
      if (is_channels_last) {
        // goihw (abcde) => gohwi (abdec)
        grouped_desc = grouped_desc.to_format(format_tag::abdec);
      }
    }

    auto this_copy = *this;
    return this_copy.set_desc(grouped_desc);
  }

  /// Recreate a param with completely different content from old one
  /// but reuse the param shell. Notice that after resize, its format
  /// is undefined
  /// legacy API for caffe2
  void resize(const dims& adims, data_type adata_type) {
    init(adims, adata_type, get_engine());
  }

  /// Return an new tensor with new shape
  tensor& reshape(const dims& adims) {
    ZENDNN_ENFORCE(has_same_volume(adims), "reshape to incompatible shape");

    // count the number of non-one dimensions
    // e.g. the actual rank of shape [1, 1, 35, 1] is one
    auto actual_rank = [](const dims& shape) {
      auto cnt = 0;
      for (auto d : shape)
        if (d > 1)
          cnt++;
      return cnt;
    };

    auto old_dims = get_dims();
    if (adims != old_dims) {
      // Since we are going to set the desc to new dims with default format,
      // we have to make sure it's already in default format. In particular,
      // tensor format does not matter if actual rank <= 1
      if (!get_desc().is_default() && actual_rank(old_dims) > 1) {
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

  inline void reorder_from(const tensor& src) {
    zendnn::reorder(src, *this)
        .execute(
            utils::stream::default_stream(), const_cast<tensor&>(src), *this);
  }

  inline void reorder_to(tensor& dst, const attr_t& aattr = attr_t()) const {
    auto pd = zendnn::reorder::primitive_desc(*this, dst, aattr);
    zendnn::reorder(pd).execute(
        utils::stream::default_stream(), const_cast<tensor&>(*this), dst);
  }

  /// Convert the tensor to public format, and f32 data type by default
  tensor to_public(void* buffer = nullptr, data_type dst_type = data_type::f32)
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

    auto dst = buffer ? tensor(dst_desc, buffer) : tensor(dst_desc);

    if (utils::one_of(
            get_data_type(), data_type::s8, data_type::u8, data_type::s32) &&
        dst_desc.get_data_type() == data_type::f32 && has_scale()) {
      auto& src_scale = get_scale();
      auto dequantize_scale =
          utils::fmap(src_scale, [](float s) { return 1.f / s; });
      auto mask =
          utils::tensor_scale_mask(src_scale.size(), get_desc().is_grouped());
      this->reorder_to(dst, {mask, dequantize_scale});
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
  void feed_from(const tensor& src, bool is_deconv_weights = false) {
    scale_t dst_scale, src_scale;
    if (has_scale() && src.has_scale()) {
      dst_scale = get_scale();
      src_scale = src.get_scale();
    } else if (has_scale()) {
      dst_scale = get_scale();
      src_scale.assign(dst_scale.size(), 1.0f);
    } else if (src.has_scale()) {
      src_scale = src.get_scale();
      dst_scale.assign(src_scale.size(), 1.0f);
    } else {
      dst_scale = ZENDNN_DEF_SCALE;
      src_scale = ZENDNN_DEF_SCALE;
    }
    ZENDNN_ENFORCE(
        dst_scale.size() == src_scale.size(), "Invalid tensor scales");
    scale_t scales(dst_scale.size());
    for (int i = 0; i < dst_scale.size(); i++) {
      scales[i] = dst_scale[i] / src_scale[i];
    }

    auto groups = 1;
    if ((groups = get_desc().g()) > 1 || (groups = src.get_desc().g()) > 1) {
      auto mask_dst = this->make_grouped_weights(groups, is_deconv_weights);
      auto mask_src = src.make_grouped_weights(groups, is_deconv_weights);
      int mask = utils::tensor_scale_mask(src_scale.size(), true);
      mask_src.reorder_to(mask_dst, {mask, scales});
    } else {
      int mask = utils::tensor_scale_mask(src_scale.size(), false);
      src.reorder_to(*this, {mask, scales});
    }
  }

  // For backward compatibility. Will be deprecated.
  void feed_from(const dims& adims, data_type adata_type, const void* array) {
    feed_from({adims, adata_type, const_cast<void*>(array)});
  }

  tensor dequantize() const {
    tensor dst(get_desc().to_type(data_type::f32));
    ZENDNN_ENFORCE(has_scale(), "Can not find scales");
    // TODO(xpz): support per-channel dequantize
    ZENDNN_ENFORCE(get_scale().size() == 1, "Incorrect scale size");
    dst.feed_from(*this);
    return dst;
  }

  // reorder src to part of this tensor
  void insert_submemory(
      const tensor& src,
      const dims& adims,
      const dims& offsets,
      const attr_t& attr = attr_t()) {
    auto view = get_desc().submemory_desc(adims, offsets);
    zendnn::reorder(
        {src.get_engine(), src.get_desc(), get_engine(), view, attr})
        .execute(
            utils::stream::default_stream(), const_cast<tensor&>(src), *this);
  }

  // reorder part of this tensor to dst
  void extract_submemory(
      tensor& dst,
      const dims& adims,
      const dims& offsets,
      const attr_t& attr = attr_t()) const {
    auto view = get_desc().submemory_desc(adims, offsets);
    zendnn::reorder(
        {get_engine(), view, dst.get_engine(), dst.get_desc(), attr})
        .execute(
            utils::stream::default_stream(), const_cast<tensor&>(*this), dst);
  }

  // simple api for extract_submemory
  tensor extract_submemory(
      const dims& adims,
      const dims& offsets,
      const attr_t& attr = attr_t()) const {
    tensor dst{adims, get_data_type(), get_engine()};
    extract_submemory(dst, adims, offsets, attr);
    return dst;
  }

  void init_workspace(const desc& desc) {
    auto workspace = new tensor(desc, get_engine());
    workspace_.reset(workspace);
  }

  /// Return extra packed tensor
  tensor& get_workspace() const {
    return *workspace_;
  }

  /// Decide wether there is an extra tensor packed in
  bool has_workspace() const {
    return workspace_ != nullptr;
  }

  /// Return the scale of this param.
  const scale_t& get_scale() const {
    return *scale_.get();
  }

  /// Set new scale into param
  void set_scale(const scale_t& ascale) {
    scale_.reset(new scale_t(ascale));
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

  /// Need reorder if current param used by non ZENDNN routines.
  // legacy API for caffe2
  inline bool need_reorder() const {
    return (!is_public_format() || get_data_type() != data_type::f32);
  }

  tensor& permute_(const std::vector<int>& permute_axes = {}) {
    return set_desc(get_desc().permute(permute_axes));
  }

  tensor permute(const std::vector<int>& permute_axes = {}) const {
    auto src_mask = *this;
    src_mask.permute_(permute_axes);
    auto dst = tensor(src_mask.get_desc().to_default_format());
    src_mask.reorder_to(dst);
    return dst;
  }

  tensor& transpose_(dim dim0, dim dim1) {
    return set_desc(get_desc().transpose(dim0, dim1));
  }

  tensor transpose(dim dim0, dim dim1) const {
    auto src_mask = *this;
    src_mask.transpose_(dim0, dim1);
    auto dst = tensor(src_mask.get_desc().to_default_format());
    src_mask.reorder_to(dst);
    return dst;
  }

  // For backward compatibility. Will be deprecated
  void transpose_from(const tensor& src, const std::vector<int>& perms = {}) {
    *this = std::move(src.permute(perms));
  }

 private:
  void reset_internal(
      const desc& adesc,
      const utils::engine& aengine,
      void* ahandle) {
    zendnn_memory_t result;
    error::wrap_c_api(
        zendnn_memory_create(&result, &adesc.data, aengine.get(), ahandle),
        "could not create a memory");
    reset(result);
  }

  inline void to_format(const desc& adesc) {
    if (get_desc() != adesc) {
      auto dst = tensor(adesc);
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
  tensor& set_desc(const desc& new_desc) {
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

  std::shared_ptr<tensor> workspace_;
  std::shared_ptr<scale_t> scale_;
  std::shared_ptr<std::vector<int32_t>> zero_point_;
  std::shared_ptr<void> buffer_;
  utils::engine eng_;
};

} // namespace zendnn
