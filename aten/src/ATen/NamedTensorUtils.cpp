#ifdef BUILD_NAMEDTENSOR

#include <ATen/NamedTensorUtils.h>
#include <bitset>
#include <sstream>

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

  const auto it = std::find_if(
      names.begin(), names.end(),
      [&dim](const Dimname& candidate) { return dim.can_refer_to(candidate); });
  TORCH_CHECK(it != names.end(),
      "Name ", dim, " not found in ", toDimnameRepr(tensor), ".");

  // Check that it can't refer to another dimension
  const auto dup = std::find_if(
      it + 1, names.end(),
      [&dim](const Dimname& candidate) { return dim.can_refer_to(candidate); });
  TORCH_CHECK(
      dup == names.end(),
      "Name ", dim, " could refer to multiple dimensions in ",
      toDimnameRepr(tensor), ". Please disambiguate by using a more ",
      "specific name like ", *it, " or ", dup, ".");
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
    DimnameList other_names) {
  // TODO(zou3519): Can improve message by checking if names are alignable and suggesting workarounds
  TORCH_CHECK(false,
      "Names ", name, " and ", other_name, " do not match positionally ",
      "from the right in names ", names, " and ", other_names, ".");
}

static void check_for_misalignment(
    const Dimname& name,
    DimnameList names,
    DimnameList other_names) {
  if (name.is_wildcard()) {
    return;
  }
  auto it = std::find_if(other_names.begin(), other_names.end(),
      [&](const Dimname& candidate) { return name.can_refer_to(candidate); });
  // TODO(zou3519): Can improve message by checking if names are alignable and suggesting workarounds
  TORCH_CHECK(it == other_names.end(),
      "Misaligned dims when attempting to broadcast dims ", names, " and dims ",
      other_names, ": dim ", name, " appears in a different position from the right "
      "across both lists");
}

// Assumption: A DimnameList can have no duplicate full names with
// the exception of wildcards
std::vector<Dimname> unify_from_right(DimnameList names, DimnameList other_names) {
  const auto wildcard = Dimname::wildcard();
  const auto size = std::max(names.size(), other_names.size());
  auto result = std::vector<Dimname>(size, wildcard);

  auto names_it = names.rbegin();
  auto other_it = other_names.rbegin();
  auto result_it = result.rbegin();
  while (names_it != names.rend() || other_it != other_names.rend()) {
    const auto& name = names_it == names.rend() ? wildcard : *names_it;
    const auto& other_name = other_it == other_names.rend() ? wildcard : *other_it;

    // TODO(zou3519): Don't support tagged names for now. They're a little weird.
    if (name.is_tagged() || other_name.is_tagged()) {
      TORCH_INTERNAL_ASSERT("unify_from_right: NYI: tagged names.");
    }

    // Step 1: Check that the names match
    const auto maybeName = unify(name, other_name);
    if (!maybeName) {
      report_positional_error(name, other_name, names, other_names);
    }
    *result_it = *maybeName;

    // Step 2: Check that the names are not misaligned
    if (!name.is_normal() || !other_name.is_normal()) {
      // Let: N = max(len(names), len(other_names))
      //      K = # of special names among names and other_names.
      // This search (including the outer loop) is O(N*K) but typically # of dims is small.
      check_for_misalignment(name, names, other_names);
      check_for_misalignment(other_name, other_names, names);
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

static std::bitset<kMaxNamedTensorDim>
compute_included_idxs(IntArrayRef excluded_idxs) {
  std::bitset<kMaxNamedTensorDim> included_idxs;
  for (auto i : excluded_idxs) {
    TORCH_INTERNAL_ASSERT(
        i <= kMaxNamedTensorDim,
        "Only tensors with up to ", kMaxNamedTensorDim, " are supported.");
    included_idxs.set(i);
  }
  included_idxs.flip();
  return included_idxs;
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
    outnames.erase(outnames.begin() + excluded_idxs[0]);
    propagate_names(result, std::move(outnames), /*validate_names=*/false);
    return;
  }

  std::vector<Dimname> outnames;
  outnames.reserve(result_dim);
  auto included_idxs = compute_included_idxs(excluded_idxs);
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

static void check_duplicate_feature_names(
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

// Let batch_dims = everything except for the last two dimensions
//     feature_dims = the last two dims of the tensor.
// We check that names of batch_dims don't overlap with the names of feature_dims.
// For example,
// Tensor[N, A, B] @ Tensor[N, A] -> Misaligned.
static void check_matmul_alignment(DimnameList self_names, DimnameList other_names) {
  auto self_batch_dims = batch_dims(self_names);
  auto self_feature_dims = feature_dims(self_names);
  auto other_batch_dims = batch_dims(other_names);
  auto other_feature_dims = feature_dims(other_names);

  for (const auto& name : self_feature_dims) {
    if (std::any_of(other_batch_dims.begin(), other_batch_dims.end(),
        [&](const Dimname& target) {
          return !target.is_wildcard() && target == name;
        })) {
      TORCH_CHECK(false,
          "Misaligned dims when batch matrix multiplying Tensor", self_names,
          " and Tensor", other_names, ": there is overlap between the feature dims ",
          "of the second tensor (", other_feature_dims,
          ") and the batch dims of the first tensor (", self_batch_dims,
          "). Please ensure batch dims are not present in the feature dims.");
    }
  }
  for (const auto& name : other_feature_dims) {
    if (std::any_of(self_batch_dims.begin(), self_batch_dims.end(),
        [&](const Dimname& target) {
          return !target.is_wildcard() && target == name;
        })) {
      TORCH_CHECK(false,
          "Misaligned dims when batch matrix multiplying Tensor", self_names,
          " and Tensor", other_names, ": there is overlap between the feature dims ",
          "of the first tensor (", self_feature_dims,
          ") and the batch dims of the second tensor (", other_batch_dims,
          "). Please ensure batch dims are not present in the feature dims.");
    }
  }
}

// Compute the outnames of torch.matmul(A, B).
static std::vector<Dimname> compute_matmul_outnames(
    DimnameList self_names,
    DimnameList other_names) {
  TORCH_CHECK(self_names.size() >= 1 && other_names.size() >= 1,
      "both arguments to matmul need to be at least 1D, but they are ",
      self_names.size(), "D and ", other_names.size(), "D");

  check_matmul_alignment(self_names, other_names);

  auto self_feature_dims = feature_dims(self_names);
  auto mm_outnames = compute_dot_product_outnames(
      self_feature_dims,
      /*tensor_dotted_dim=*/self_feature_dims.size() - 1, // last dim
      feature_dims(other_names),
      /*other_dotted_dim=*/0);

  std::vector<Dimname> result;
  auto self_batch_dims = batch_dims(self_names);
  auto other_batch_dims = batch_dims(other_names);
  if (self_batch_dims.empty() && other_batch_dims.empty()) {
    result = mm_outnames;
  } else {
    result = unify_from_right(batch_dims(self_names), batch_dims(other_names));
    result.insert(result.end(), mm_outnames.begin(), mm_outnames.end());
  }

  check_duplicate_feature_names(self_names, other_names, result);
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
