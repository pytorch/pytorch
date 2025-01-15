#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorNames.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <c10/util/irange.h>

#include <bitset>
#include <sstream>

namespace at {

#ifndef STRIP_ERROR_MESSAGES
// Returns "Tensor['N', 'C', 'H', 'W']" for a tensor with names ('N', 'C', 'H', 'W').
static std::string toDimnameRepr(const Tensor& tensor) {
  std::ostringstream os;
  os << "Tensor" << tensor.names();
  return os.str();
}
#endif

int64_t dimname_to_position(const Tensor& tensor, Dimname dim) {
  TORCH_CHECK(dim.type() != NameType::WILDCARD,
      "Please look up dimensions by name, got: name = None.");
  TORCH_CHECK(tensor.has_names(),
      "Name ", dim, " not found in ", toDimnameRepr(tensor), ".");
  const auto names = tensor.names();

  const auto it = std::find(names.begin(), names.end(), dim);
  TORCH_CHECK(it != names.end(),
      "Name ", dim, " not found in ", toDimnameRepr(tensor), ".");

  return std::distance(names.begin(), it);
}

std::vector<int64_t> dimnames_to_positions(const Tensor& tensor, DimnameList dims) {
  std::vector<int64_t> result;
  result.reserve(dims.size());
  for (const auto& name : dims) {
    result.push_back(dimname_to_position(tensor, name));
  }
  return result;
}

[[noreturn]] static void report_positional_error(
    const Dimname& name,
    const Dimname& other_name,
    DimnameList names,
    DimnameList other_names,
    const char* action) {
  // TODO(zou3519): Can improve message by checking if names are alignable and suggesting workarounds
  TORCH_CHECK(false,
      "Error when attempting to ", action, " dims ", names, " and dims ",
      other_names, ": dim ", name, " and dim ", other_name, " are at the same position "
      "from the right but do not match.")
}

static void check_for_misalignment(
    const Dimname& name,
    DimnameList names,
    DimnameList other_names,
    const char* action) {
  if (name.isWildcard()) {
    return;
  }
  auto it = std::find(other_names.begin(), other_names.end(), name);
  // TODO(zou3519): Can improve message by checking if names are alignable and suggesting workarounds
  TORCH_CHECK(it == other_names.end(),
      "Misaligned dims when attempting to ", action, " dims ", names, " and dims ",
      other_names, ": dim ", name, " appears in a different position from the right "
      "across both lists.");
}

// Assumption: A DimnameList can have no duplicate full names with
// the exception of wildcards
std::vector<Dimname> unify_from_right(
    DimnameList names,
    DimnameList other_names,
    const char* action) {
  const auto wildcard = Dimname::wildcard();
  const auto size = std::max(names.size(), other_names.size());
  auto result = std::vector<Dimname>(size, wildcard);

  auto names_it = names.rbegin();
  auto other_it = other_names.rbegin();
  auto result_it = result.rbegin();
  while (names_it != names.rend() || other_it != other_names.rend()) {
    const auto& name = names_it == names.rend() ? wildcard : *names_it;
    const auto& other_name = other_it == other_names.rend() ? wildcard : *other_it;

    // Step 1: Check that the names match
    const auto maybeName = name.unify(other_name);
    if (!maybeName) {
      report_positional_error(name, other_name, names, other_names, action);
    }
    *result_it = *maybeName;

    // Step 2: Check that the names are not misaligned
    if (!name.isBasic() || !other_name.isBasic()) {
      // Let: N = max(len(names), len(other_names))
      //      K = # of special names among names and other_names.
      // This search (including the outer loop) is O(N*K) but typically # of dims is small.
      check_for_misalignment(name, names, other_names, action);
      check_for_misalignment(other_name, other_names, names, action);
    }

    if (names_it != names.rend()) {
      ++names_it;
    }
    if (other_it != other_names.rend()) {
      ++other_it;
    }
    ++result_it;
  }
  return result;
}

namespace namedinference {

static std::bitset<dim_bitset_size>
compute_included_idxs(IntArrayRef excluded_idxs, int64_t ndims) {
  auto result = dim_list_to_bitset(excluded_idxs, ndims);
  result.flip();
  return result;
}

static void assert_names_equal(DimnameList a, DimnameList b) {
  TORCH_CHECK(a == b,
      "Name mismatch: specified out tensor with names ", a,
      " are not the same as the computed output names ", b,
      ". Please rename the out tensor's dims with `Tensor.rename`.");
}

const Tensor& propagate_names_if_present_and_nonempty(const Tensor& result,
    std::optional<DimnameList> maybe_names,
    bool validate_names) {
  auto maybe_name_list = maybe_names.value_or(at::ArrayRef<Dimname>{});
  propagate_names_if_nonempty(result.unsafeGetTensorImpl(), maybe_name_list, validate_names);
  return result;
}

const Tensor& propagate_names_if_nonempty(const Tensor& result,
    DimnameList maybe_names,
    bool validate_names) {
  propagate_names_if_nonempty(result.unsafeGetTensorImpl(), maybe_names, validate_names);
  return result;
}

TensorImpl* propagate_names_if_nonempty(TensorImpl* result,
    DimnameList maybe_names,
    bool validate_names) {
  if (maybe_names.empty()) {
    return result;
  }
  return propagate_names(result, maybe_names, validate_names);
}

const Tensor& propagate_names(const Tensor& result, DimnameList names, bool validate_names) {
  propagate_names(result.unsafeGetTensorImpl(), names, validate_names);
  return result;
}

TensorImpl* propagate_names(TensorImpl* result, DimnameList names, bool validate_names) {
  if (result->dim() > 0) {
    TORCH_INTERNAL_ASSERT(
        !names.empty(),
        "propagate_names: passed in empty names to propagate to result with",
        " shape ", result->sizes(), ". Empty names means that name inference did",
        "not occur; use `propagate_names_if_nonempty` instead of `propagate_names`.");
  }
  if (!impl::has_names(result)) {
    impl::internal_set_names_inplace(result, names, validate_names);
  } else {
    assert_names_equal(impl::get_names(result), names);
  }
  return result;
}

void propagate_names_except(const Tensor& result, const Tensor& src, IntArrayRef excluded_idxs) {
  if (!result.has_names() && !src.has_names()) {
    return;
  }
  const auto src_names = src.names();
  const auto result_dim = static_cast<int64_t>(result.dim());
  const auto src_dim = static_cast<int64_t>(src_names.size());
  const auto excluded_dim = static_cast<int64_t>(excluded_idxs.size());
  TORCH_INTERNAL_ASSERT(src_dim - excluded_dim == result_dim);

  // fast path
  if (excluded_idxs.size() == 1) {
    std::vector<Dimname> outnames = src_names.vec();
    outnames.erase(outnames.begin() + maybe_wrap_dim(excluded_idxs[0], src_dim));
    propagate_names(result, outnames);
    return;
  }

  std::vector<Dimname> outnames;
  outnames.reserve(result_dim);
  auto included_idxs = compute_included_idxs(excluded_idxs, src_dim);
  for (const auto dim : c10::irange(src_dim)) {
    if (included_idxs[dim]) {
      outnames.push_back(src_names[dim]);
    }
  }
  propagate_names(result, outnames);
}

void propagate_names_for_reduction(const Tensor& result, const Tensor& src, IntArrayRef reduced_dims, bool keepdim) {
  if (keepdim) {
    propagate_names(result, src);
    return;
  }
  // This actually means "full reduction"
  if (reduced_dims.empty()) {
    return;
  }
  propagate_names_except(result, src, reduced_dims);
}

void propagate_names(const Tensor& result, const Tensor& src) {
  propagate_names(result.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
}

void propagate_names(TensorImpl* result, TensorImpl* src) {
  if (result == src) {
    return;
  }
  if (!impl::has_names(result) && !impl::has_names(src)) {
    return;
  }
  propagate_names(result, impl::get_names(src));
}

std::vector<Dimname> compute_squeeze_outnames(const Tensor& tensor) {
  if (!tensor.has_names()) {
    return {};
  }
  std::vector<Dimname> outnames;
  auto tensor_names = tensor.names();
  for (const auto d : c10::irange(tensor.dim())) {
    if (tensor.sym_sizes()[d] != 1) {
      outnames.push_back(tensor_names[d]);
    }
  }
  return outnames;
}

std::vector<Dimname> compute_squeeze_outnames(const Tensor& tensor, std::bitset<dim_bitset_size> dims) {
  if (!tensor.has_names()) {
    return {};
  }
  std::vector<Dimname> outnames;
  auto tensor_names = tensor.names();
  for (const auto d : c10::irange(tensor.dim())) {
    if (!dims.test(d) || tensor.sym_sizes()[d] != 1) {
      outnames.push_back(tensor_names[d]);
    }
  }
  return outnames;
}

std::vector<Dimname> compute_diagonal_outnames(
    const Tensor& tensor,
    int64_t dim1,
    int64_t dim2) {
  if (!tensor.has_names()) {
    return {};
  }
  std::vector<Dimname> outnames;
  auto tensor_names = tensor.names();
  for (const auto d : c10::irange(tensor.dim())) {
    if (d == dim1 || d == dim2) {
      continue;
    }
    outnames.push_back(tensor_names[d]);
  }
  outnames.push_back(Dimname::wildcard());
  return outnames;
}

static void check_feature_names_are_distinct(
    DimnameList self_names,
    DimnameList other_names,
    const DimnameList& outnames) {
  if (self_names.size() < 2 || other_names.size() < 2) {
    // There are less than 2 feature dims in outnames so there is nothing to check
    return;
  }
  auto feature0 = outnames[outnames.size() - 2];
  auto feature1 = outnames[outnames.size() - 1];
  TORCH_CHECK(
    feature0 == Dimname::wildcard() || feature0 != feature1,
    "Matrix multiplying Tensor", self_names,
    " with Tensor", other_names,
    " would produce output tensor with duplicate names ",
    outnames,
    ". Please rename the input tensors with `Tensor.rename` to prevent this.");
}

static int64_t num_batch_dims(DimnameList names) {
  if (names.size() <= 2) {
    return 0;
  }
  return static_cast<int64_t>(names.size() - 2);
}

static std::vector<Dimname> compute_matmul_outnames(
    DimnameList self_names,
    DimnameList other_names) {
  TORCH_CHECK(!self_names.empty() && !other_names.empty(),
      "both arguments to matmul need to be at least 1D, but they are ",
      self_names.size(), "D and ", other_names.size(), "D");

  // matmul performs a batch matrix multiply between self and other, each of which
  // can either be:
  // - a batches of matrices (if dim > 2)
  // - a matrix (if dim == 2)
  // - a vector (if dim == 1)
  //
  // To compute output names, we unify the batch dimensions because those are
  // broadcastable to get the output batch dimensions.
  //
  // After that, we append some names that are equal to the result of the matmul
  // without batch dimensions. Those names are computed by removing the names
  // of the dimensions that were contracted away. We always contract the
  // last dim of the first tensor with the first feature dimension of the second.

  // Get the output's batch dimension names
  auto wrapped_self_names = TensorNames(self_names, 0, num_batch_dims(self_names));
  const auto wrapped_other_names = TensorNames(other_names, 0, num_batch_dims(other_names));
  auto& working_names = wrapped_self_names.unifyFromRightInplace(wrapped_other_names, "matmul");

  // Append the result of each individual (non-batched) matmul.
  // If either of self or other have dim 1, that means they are a vector. Vectors get
  // completely contracted away during matmul so we don't take any names from them.
  if (self_names.size() >= 2) {
    working_names.append(TensorName(self_names, -2));
  }
  if (other_names.size() >= 2) {
    working_names.append(TensorName(other_names, -1));
  }
  auto result = working_names.toDimnameVec();

  check_feature_names_are_distinct(self_names, other_names, result);
  return result;
}

std::vector<Dimname> propagate_names_for_addmv(
    const Tensor& mat,
    const Tensor& vec,
    const Tensor& bias) {
  if (!mat.has_names() &&
      !vec.has_names() && !bias.has_names()) {
    return std::vector<Dimname>{};
  }
  auto mv_outnames = compute_matmul_outnames(mat.names(), vec.names());
  return unify_from_right(mv_outnames, bias.names());
}

std::vector<Dimname> propagate_names_for_addmm(
    const Tensor& m1,
    const Tensor& m2,
    const Tensor& bias) {
  if (!m1.has_names() && !m2.has_names() &&
      !bias.has_names()) {
    return std::vector<Dimname>{};
  }

  auto mm_outnames = compute_matmul_outnames(m1.names(), m2.names());
  return unify_from_right(mm_outnames, bias.names());
}

void check_names_for_dot(
    TensorImpl* vec1,
    TensorImpl* vec2) {
  if (!impl::has_names(vec1) && !impl::has_names(vec2)) {
    return;
  }
  compute_matmul_outnames(impl::get_names(vec1), impl::get_names(vec2));
}

// expand adds new None dimensions. This is consistent with name inference
// rules for binary ops that expect the named dims to line up positionally
// from the right. i.e.,
// Tensor[H, W].expand(3, 3, 3, 3) -> Tensor[None, None, H, W]
void propagate_names_for_expand(const Tensor& result, const Tensor& self) {
  if (!self.has_names()) {
    return;
  }
  auto result_dim = result.dim();
  if (self.dim() == result_dim) {
    propagate_names(result, self);
    return;
  }
  std::vector<Dimname> outnames(result_dim, Dimname::wildcard());
  auto const names = self.names();
  std::copy( names.begin(), names.end(), outnames.begin() + result_dim - self.dim());
  propagate_names(result, outnames);
}

std::vector<Dimname> compute_broadcast_outnames(
    const Tensor& self,
    const Tensor& other) {
  if (!self.has_names() && !other.has_names()) {
    return {};
  }
  return unify_from_right(self.names(), other.names());
}

std::vector<Dimname> broadcast_to_outnames(
    const Tensor& tensor,
    const Tensor& reference_tensor,
    const char* op_name) {
  if (!tensor.has_names() && !reference_tensor.has_names()) {
    return {};
  }
  auto reference_names = reference_tensor.names();
  auto tensor_names = tensor.names();
  TORCH_CHECK(
      reference_names.size() >= tensor_names.size(),
      op_name, ": attempted to broadcast Tensor", tensor_names, " to Tensor",
      reference_names, " but the number of dims (", tensor_names.size(),
      ") must be less than or equal to the number of dims in the tensor (",
      reference_names.size(), ")");
  return unify_from_right(reference_names, tensor_names);
}

std::vector<Dimname> compute_cat_outnames(const MaterializedITensorListRef& tensors) {
  if (!at::has_names(tensors)) {
    return {};
  }
  std::vector<Dimname> result;
  for (const Tensor& tensor : tensors) {
    const auto tensor_names = tensor.names();
    TORCH_CHECK(!tensor_names.empty(), "zero-dimensional tensor cannot be concatenated");
    TORCH_CHECK(result.empty() || tensor_names.size() == result.size(),
        "Tensors must have same number of dimensions: got ", result.size(),
        " and ", tensor_names.size());
    result = unify_from_right(result, tensor_names, "cat");
  }
  return result;
}

std::vector<Dimname> compute_matmul_outnames(
    const Tensor& self,
    const Tensor& other) {
  if (!self.has_names() && !other.has_names()) {
    return {};
  }
  return compute_matmul_outnames(self.names(), other.names());
}

std::vector<Dimname> compute_cdist_outnames(
    const Tensor& self,
    const Tensor& other) {
  if (!self.has_names() && !other.has_names()) {
    return {};
  }
  const auto self_names = self.names();
  const auto other_names = other.names();

  auto self_batch = TensorNames(self_names, 0, num_batch_dims(self_names));
  const auto other_batch = TensorNames(other_names, 0, num_batch_dims(other_names));

  auto& result = self_batch.unifyFromRightInplace(other_batch, "cdist");

  // cdist treats self and other like batches of M x D and N X D tensors, respectively.
  // It computes the pairwise distance between each of the M vectors (of size D)
  // in `self` and each of the N vectors in `other`, returning a batch of M x N
  // distance values. We propagate the names of the dimension of size M (in self)
  // and the dimension of size N (in other), both of which are second-from-last.
  result.append(TensorName(self_names, -2));
  result.append(TensorName(other_names, -2));
  result.checkUnique("cdist");

  return result.toDimnameVec();
}

std::vector<Dimname> compute_bmm_outnames(
    const Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  if (!result.has_names() && !self.has_names() && !other.has_names()) {
    return {};
  }
  return compute_matmul_outnames(self.names(), other.names());
}

std::vector<Dimname> compute_baddbmm_outnames(
    const Tensor& result,
    const Tensor& self,
    const Tensor& other,
    const Tensor& bias) {
  if (!result.has_names() && !self.has_names()
    && !other.has_names() && !bias.has_names()) {
    return {};
  }
  auto bmm_names = compute_matmul_outnames(self.names(), other.names());
  auto baddbmm_names = unify_from_right(bias.names(), bmm_names);
  return baddbmm_names;
}

bool are_names_equal(TensorImpl* self, TensorImpl* other) {
  if (!impl::has_names(self) && !impl::has_names(other)) {
    return true;
  }
  return impl::get_names(self) == impl::get_names(other);
}

} // namespace namedinference
} // namespace at
