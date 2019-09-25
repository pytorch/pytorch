#include <ATen/NamedTensorUtils.h>
#include <ATen/core/EnableNamedTensor.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <bitset>
#include <sstream>

#ifdef BUILD_NAMEDTENSOR
namespace at {

// Returns "Tensor['N', 'C', 'H', 'W']" for a tensor with names ('N', 'C', 'H', 'W').
static std::string toDimnameRepr(const Tensor& tensor) {
  std::ostringstream os;
  os << "Tensor" << tensor.names();
  return os.str();
}

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

static void report_positional_error(
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
      "across both lists");
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
      ". Please rename the out tensor's dimensions.");
}

void propagate_names(TensorImpl* result, optional<DimnameList> names) {
  if (!impl::has_names(result) && !names.has_value()) {
    return;
  }
  if (!impl::has_names(result)) {
    impl::internal_set_names_inplace(result, names);
    return;
  }
  assert_names_equal(
      impl::get_names(result),
      names.value_or(default_names(result->dim())));
}

void propagate_names(
    Tensor& result,
    optional<std::vector<Dimname>>&& maybe_names,
    bool validate_names) {
  propagate_names(result.unsafeGetTensorImpl(), std::move(maybe_names), validate_names);
}

void propagate_names(
    TensorImpl* result,
    optional<std::vector<Dimname>>&& maybe_names,
    bool validate_names) {
  if (!maybe_names) {
    propagate_names(result, nullopt);
    return;
  }
  propagate_names(result, std::move(maybe_names.value()), validate_names);
}

void propagate_names(TensorImpl* result, std::vector<Dimname>&& names, bool validate_names) {
  if (!impl::has_names(result)) {
    impl::internal_set_names_inplace(result, std::move(names), validate_names);
    return;
  }
  assert_names_equal(impl::get_names(result), names);
}

void propagate_names(Tensor& result, optional<DimnameList> names) {
  propagate_names(result.unsafeGetTensorImpl(), names);
}

void propagate_names(Tensor& result, std::vector<Dimname>&& names, bool validate_names) {
  propagate_names(result.unsafeGetTensorImpl(), std::move(names), validate_names);
}

void propagate_names_except(Tensor& result, const Tensor& src, IntArrayRef excluded_idxs) {
  if (!result.has_names() && !src.has_names()) {
    return;
  }
  auto src_names = src.names();
  auto result_dim = result.dim();
  auto src_dim = src_names.size();
  TORCH_INTERNAL_ASSERT(src_dim - excluded_idxs.size() == result_dim);

  // fast path
  if (excluded_idxs.size() == 1) {
    std::vector<Dimname> outnames = src_names.vec();
    outnames.erase(outnames.begin() + maybe_wrap_dim(excluded_idxs[0], src_dim));
    propagate_names(result, std::move(outnames), /*validate_names=*/false);
    return;
  }

  std::vector<Dimname> outnames;
  outnames.reserve(result_dim);
  auto included_idxs = compute_included_idxs(excluded_idxs, src_dim);
  for (size_t dim = 0; dim < src_dim; ++dim) {
    if (included_idxs[dim]) {
      outnames.push_back(src_names[dim]);
    }
  }
  propagate_names(result, std::move(outnames), /*validate_names=*/false);
}

void propagate_names_for_reduction(Tensor& result, const Tensor& src, IntArrayRef reduced_dims, bool keepdim) {
  if (keepdim) {
    propagate_names(result, src);
    return;
  }
  // This actually means "full reduction"
  if (reduced_dims.size() == 0) {
    return;
  }
  propagate_names_except(result, src, reduced_dims);
}

void propagate_names(Tensor& result, const Tensor& src) {
  propagate_names(result.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
}

void propagate_names(TensorImpl* result, TensorImpl* src) {
  if (result == src) {
    return;
  }
  propagate_names(result, impl::get_opt_names(src));
}

void propagate_names_for_copy(Tensor& result, const Tensor& src) {
  if (!result.has_names() && !src.has_names()) {
    return;
  }
  auto outnames = unify_from_right(result.names(), src.names());
  propagate_names(result, std::move(outnames), /*validate_names=*/false);
}

// tensor_dotted_dim and other_dotted_dim are the dimensions of the two
// tensors that we contract together. Usually other_dotted_dim is 0
// and tensor_dotted_dim is the last dim of tensor, but there are some special
// cases like einsum and tensordot where one can contract arbitrary dims.
static std::vector<Dimname> compute_dot_product_outnames(
    DimnameList tensor_names,
    int64_t tensor_dotted_dim,
    DimnameList other_names,
    int64_t other_dotted_dim) {
  int64_t num_outnames = tensor_names.size() + other_names.size() - 2;
  if (num_outnames == 0) {
    return {};
  }
  std::vector<Dimname> outnames(num_outnames, Dimname::wildcard());
  int64_t index = 0;
  for (int64_t j = 0; j < tensor_names.size(); ++j) {
    if (j == tensor_dotted_dim) continue;
    outnames[index++] = tensor_names[j];
  }
  for (int64_t j = 0; j < other_names.size(); ++j) {
    if (j == other_dotted_dim) continue;
    outnames[index++] = other_names[j];
  }
  return outnames;
}

static void check_feature_names_are_distinct(
    DimnameList self_names,
    DimnameList other_names,
    DimnameList outnames) {
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
    ". Please rename the input tensors to prevent this.");
}

static DimnameList batch_dims(DimnameList names) {
  if (names.size() <= 2) {
    return {};
  }
  return DimnameList(names.begin(), names.end() - 2);
}

static DimnameList feature_dims(DimnameList names) {
  if (names.size() <= 2) {
    return names;
  }
  return DimnameList(names.end() - 2, 2);
}

static bool are_distinct(DimnameList batch_dims, DimnameList feature_dims) {
  for (const auto& target : feature_dims) {
    if (target.isWildcard()) {
      continue;
    }
    if (std::any_of(batch_dims.begin(), batch_dims.end(),
          [&](const Dimname& dim) { return target == dim; })) {
      return false;
    }
  }
  return true;
}

// Let batch_dims = everything except for the last two dimensions
//     feature_dims = the last two dims of the tensor.
// This function checks that names of batch_dims of one tensor are distinct
// from the feature dimensions of another tensor.
//
// For example,
// Tensor[N, A, B] @ Tensor[None, N, A] -> Misaligned because N (a batch dim of
// the first tensor) appears in a non-batch dimension in the second tensor.
//
// Tensor[N, A, B1] @ Tensor[N, B2] -> Misaligned! The user may have intended
// to batch multiply matrices of size [A, B1] with vectors of size [B2]
// but instead matmul treats it as batch multiplying matrices of size [A, B1]
// with matrices of size [N, B2]. In this case, we're able to warn the user.
static void check_batch_and_feature_dims_are_distinct(
    DimnameList self_names,
    DimnameList other_names) {
  auto self_batch_dims = batch_dims(self_names);
  auto self_feature_dims = feature_dims(self_names);
  auto other_batch_dims = batch_dims(other_names);
  auto other_feature_dims = feature_dims(other_names);

  TORCH_CHECK(are_distinct(self_batch_dims, other_feature_dims),
      "Misaligned dims when batch matrix multiplying Tensor", self_names,
      " and Tensor", other_names, ": there is overlap between the feature dims ",
      "of the second tensor (", other_feature_dims,
      ") and the batch dims of the first tensor (", self_batch_dims,
      "). Please ensure batch dims are not present in the feature dims.");

  TORCH_CHECK(are_distinct(other_batch_dims, self_feature_dims),
      "Misaligned dims when batch matrix multiplying Tensor", self_names,
      " and Tensor", other_names, ": there is overlap between the feature dims ",
      "of the first tensor (", self_feature_dims,
      ") and the batch dims of the second tensor (", other_batch_dims,
      "). Please ensure batch dims are not present in the feature dims.");
}

// Compute the outnames of torch.matmul(A, B).
static std::vector<Dimname> compute_matmul_outnames(
    DimnameList self_names,
    DimnameList other_names) {
  TORCH_CHECK(self_names.size() >= 1 && other_names.size() >= 1,
      "both arguments to matmul need to be at least 1D, but they are ",
      self_names.size(), "D and ", other_names.size(), "D");

  check_batch_and_feature_dims_are_distinct(self_names, other_names);

  // The approach is to
  // (1) compute the outnames of the matrix multiplication on the feature dims.
  // (2) compute the outnames of the batch dimensions, after broadcasting
  // (3) concatenate the batch outnames and matrix multiply outnames.

  // Step 1: Compute outnames of matrix multiplication
  // Let N >= 2.
  // if ND @ ND we matrix multiply the last two dimensions of both tensors.
  // if ND @ 1D, it is a batch matrix (last two dims of first tensor) - vector multiply
  // if 1D @ ND, it is a batch vector - matrix (last two dims of second tensor) multiply
  // In all cases, we are contracting the last dimension of the first tensor
  // with the first non-batch dimension of the second tensor.
  auto self_feature_dims = feature_dims(self_names);
  auto mm_outnames = compute_dot_product_outnames(
      self_feature_dims,
      /*tensor_dotted_dim=*/self_feature_dims.size() - 1, // last dim
      feature_dims(other_names),
      /*other_dotted_dim=*/0);

  // Step 2: Figure out the outnames of the batch dimensions.
  std::vector<Dimname> result;
  auto self_batch_dims = batch_dims(self_names);
  auto other_batch_dims = batch_dims(other_names);
  if (self_batch_dims.empty() && other_batch_dims.empty()) {
    result = mm_outnames;
  } else {
    result = unify_from_right(self_batch_dims, other_batch_dims);
    result.insert(result.end(), mm_outnames.begin(), mm_outnames.end());
  }

  check_feature_names_are_distinct(self_names, other_names, result);
  return result;
}

void propagate_names_for_addmv(
    TensorImpl* result,
    TensorImpl* mat,
    TensorImpl* vec,
    TensorImpl* bias) {
  if (!impl::has_names(result) && !impl::has_names(mat) &&
      !impl::has_names(vec) && !impl::has_names(bias)) {
    return;
  }
  auto mv_outnames = compute_matmul_outnames(impl::get_names(mat), impl::get_names(vec));
  auto add_outnames = unify_from_right(mv_outnames, impl::get_names(bias));
  propagate_names(result, std::move(add_outnames), /*validate_names=*/false);
}

void propagate_names_for_addmm(
    TensorImpl* result,
    TensorImpl* m1,
    TensorImpl* m2,
    TensorImpl* bias) {
  if (!impl::has_names(m1) && !impl::has_names(m2) &&
      !impl::has_names(bias) && !impl::has_names(result)) {
    return;
  }
  auto mm_outnames = compute_matmul_outnames(impl::get_names(m1), impl::get_names(m2));
  auto add_outnames = unify_from_right(mm_outnames, impl::get_names(bias));
  propagate_names(result, std::move(add_outnames), /*validate_names=*/false);
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
void propagate_names_for_expand(Tensor& result, const Tensor& self) {
  if (!self.has_names()) {
    return;
  }
  auto result_dim = result.dim();
  if (self.dim() == result_dim) {
    propagate_names(result, self);
    return;
  }
  std::vector<Dimname> outnames(result_dim, Dimname::wildcard());
  std::copy(
      self.opt_names()->begin(),
      self.opt_names()->end(),
      outnames.begin() + result_dim - self.dim());
  propagate_names(result, std::move(outnames), /*validate_names=*/false);
}

optional<std::vector<Dimname>> compute_broadcast_outnames(
    const Tensor& self,
    const Tensor& other) {
  if (!self.has_names() && !other.has_names()) {
    return nullopt;
  }
  return unify_from_right(self.names(), other.names());
}

optional<std::vector<Dimname>> broadcast_to_outnames(
    const Tensor& tensor,
    const Tensor& reference_tensor,
    const char* op_name) {
  if (!tensor.has_names() && !reference_tensor.has_names()) {
    return nullopt;
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

optional<std::vector<Dimname>> compute_cat_outnames(TensorList tensors) {
  if (!at::has_names(tensors)) {
    return nullopt;
  }
  std::vector<Dimname> result;
  for (const auto& tensor : tensors) {
    const auto tensor_names = tensor.names();
    TORCH_CHECK(tensor_names.size() > 0, "zero-dimensional tensor cannot be concatenated");
    TORCH_CHECK(result.empty() || tensor_names.size() == result.size(),
        "Tensors must have same number of dimensions: got ", result.size(),
        " and ", tensor_names.size());
    result = unify_from_right(result, tensor_names, "cat");
  }
  return result;
}

optional<std::vector<Dimname>> compute_matmul_outnames(
    const Tensor& self,
    const Tensor& other) {
  if (!self.has_names() && !other.has_names()) {
    return nullopt;
  }
  return compute_matmul_outnames(self.names(), other.names());
}

optional<std::vector<Dimname>> compute_bmm_outnames(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  if (!result.has_names() && !self.has_names() && !other.has_names()) {
    return nullopt;
  }
  return compute_matmul_outnames(self.names(), other.names());
}

optional<std::vector<Dimname>> compute_baddbmm_outnames(
    TensorImpl* result,
    TensorImpl* batch1,
    TensorImpl* batch2,
    TensorImpl* bias) {
  if (!impl::has_names(result) && !impl::has_names(batch1) &&
      !impl::has_names(batch2) && !impl::has_names(bias)) {
    return nullopt;
  }
  auto bmm_names = compute_matmul_outnames(
      impl::get_names(batch1), impl::get_names(batch2));
  auto baddbmm_names = unify_from_right(impl::get_names(bias), bmm_names);
  return baddbmm_names;
}

} // namespace namedinference
} // namespace at
#endif
