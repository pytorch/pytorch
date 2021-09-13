#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>
#include <c10/util/MaybeOwned.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace at { namespace native {

Tensor linear(const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
    : c10::MaybeOwned<Tensor>::owned(c10::in_place);

  if (input.is_mkldnn()) {
    return at::mkldnn_linear(input, weight, *bias);
  }
#if defined(C10_MOBILE)
  if (xnnpack::use_linear(input, weight, *bias)) {
    return xnnpack::linear(input, weight, *bias);
  }
#endif
  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::addmm(*bias, input, weight.t());
  }
  auto output = at::matmul(input, weight.t());
  if (bias->defined()) {
    output.add_(*bias);
  }
  return output;
}

Tensor& linear_out(const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt, Tensor& output) {
  TORCH_CHECK(!input.is_mkldnn(), "linear doesn't support out for MKLDNN tensors");
  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
              ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
              : c10::MaybeOwned<Tensor>::owned(c10::in_place);

  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::addmm_out(output, *bias, input, weight.t());
  }
  output = at::matmul_out(output, input, weight.t());
  if (bias->defined()) {
    output.add_(*bias);
  }
  return output;
}

// sumproduct_pair computes `(left*right).sum(sumdims)` by means of permutation and
// batch matrix multiplication
// its main purpose is to provide a pairwise reduction for einsum
static Tensor sumproduct_pair(const Tensor& left_, const Tensor& right_, IntArrayRef sum_dims_, bool keepdim) {
  // assumes that tensors have been pre-unsqueezed (so that all dimensions match - after broadcasting)
  // but makes no other assumptions on the order of dimensions
  TORCH_CHECK(left_.dim()==right_.dim(), "number of dimensions must match");
  if (sum_dims_.size() == 0)
    return at::mul(left_, right_);
  int64_t dim = left_.dim();
  auto sum_dims = at::dim_list_to_bitset(sum_dims_, dim);
  // dimensions that will be part of the output (i.e. not summed over) in three vectors
  // dims in lro appear in left, right and output, similarly lo: left and output, ro: right and output
  // also the sizes are kept track of for reshaping
  std::vector<int64_t> lro, lo, ro;
  int64_t lro_size = 1, lo_size = 1, ro_size = 1, sum_size = 1;
  Tensor left = left_;
  Tensor right = right_;
  for (const auto i : c10::irange(dim)) {
    auto sl = left.size(i)!=1;
    auto sr = right.size(i)!=1;
    if (sum_dims[i]) { // first dimensions that will be summed over after multiplication
      if (sl && sr) {  // dimensions nontrivially in both left and right must be of the same size
        TORCH_CHECK(left.size(i)==right.size(i), "non-broadcast dimensions must match");
        sum_size *= left.size(i);
      } else if (sl) { // if it is only in one of left and right, we can sum right away
        left = left.sum(i, true);
      } else if (sr) {
        right = right.sum(i, true);
      }
    } else if (sl && sr) { // now deal with dimensions  dimensions that will be in the output
      // dimensions nontrivially in both left and right must be of the same size
      TORCH_CHECK(left.size(i)==right.size(i), "non-broadcast dimensions must match");
      lro.push_back(i);
      lro_size *= left.size(i);
    } else if (sl) { // keep track of dimensions appearing only once
      lo.push_back(i);
      lo_size *= left.size(i);
    } else {
      ro.push_back(i);
      ro_size *= right.size(i);
    }
  }
  // we now work with the following permutations / shapes.
  // the pipeline is permute inputs -> reshape inputs -> batch matrix mul -> reshape(view) output -> permute output
  // output: "lro, lo, 1-for-summed-dims, ro" with orgiginal shape dimensions
  // left:   "lro, lo, summed" permuted with lpermutation and the three flattened
  // right:  "lro, summed, ro" permuted with rpermutation and the three flattened
  // then the permuted output is a view of bmm(left, right)
  // finally, opermutation reverts the permutation to the original order of dimensions
  std::vector<int64_t> out_size;
  // NOLINTNEXTLINE(performance-inefficient-vector-operation)
  for (auto& d : lro) out_size.push_back(left.size(d));
  for (auto& d : lo) out_size.push_back(left.size(d));
  for (auto& d : sum_dims_) { out_size.push_back(1); (void)(d); }; // avoid warining about not using d
  for (auto& d : ro) out_size.push_back(right.size(d));

  std::vector<int64_t> lpermutation(lro);
  lpermutation.insert(lpermutation.end(), lo.begin(), lo.end());
  lpermutation.insert(lpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  lpermutation.insert(lpermutation.end(), ro.begin(), ro.end());

  std::vector<int64_t> rpermutation(lro);
  rpermutation.insert(rpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  rpermutation.insert(rpermutation.end(), ro.begin(), ro.end());
  rpermutation.insert(rpermutation.end(), lo.begin(), lo.end());

  std::vector<int64_t> opermutation(lro.size()+lo.size()+sum_dims_.size()+ro.size(), -1);
  {
    int64_t i = 0;

    for (auto it = lro.cbegin(); it != lro.cend(); i++, it++) {
      opermutation[*it] = i;
    }
    for (auto it = lo.cbegin(); it != lo.cend(); i++, it++) {
      opermutation[*it] = i;
    }
    for (auto it = sum_dims_.cbegin(); it != sum_dims_.cend(); i++, it++) {
      opermutation[*it] = i;
    }
    for (auto it = ro.cbegin(); it != ro.cend(); i++, it++) {
      opermutation[*it] = i;
    }
  }

  // now we can execute the operations above
  left = left.permute(lpermutation).reshape({lro_size, lo_size, sum_size});
  right = right.permute(rpermutation).reshape({lro_size, sum_size, ro_size});
  Tensor result = at::bmm(left, right);
  result = result.view(out_size).permute(opermutation);

  // finally squeeze summed dimensions if desired
  if (! keepdim) {
    auto sizes = result.sizes().vec();
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    for (int i = dim-1; i>=0; i--) {
      if (sum_dims[i]) {
        sizes.erase(sizes.begin() + i);
      }
    }
    result = result.view(sizes);
  }
  return result;
}

Tensor einsum(
    string_view equation,
    TensorList operands,
    optional<IntArrayRef> optimize) {
  TORCH_CHECK(!operands.empty(), "einsum(): must provide at least one operand");
  const auto num_ops = operands.size();

  if (optimize.has_value()) {
    const auto path_size = num_ops == 1 ? 1 : (num_ops - 1) * 2;
    TORCH_CHECK(
        optimize->size() == path_size,
        "einsum(): expected contraction path given in optimize parameter to have size ",
        path_size,
        " but got ",
        optimize->size());
  }

  // Labels must be in range [A-Za-z]
  constexpr uint8_t NUM_OF_LETTERS = 'z' - 'a' + 1;
  constexpr uint8_t TOTAL_LABELS = NUM_OF_LETTERS * 2;

  // Used to identify an ellipsis
  constexpr uint8_t ELLIPSIS = TOTAL_LABELS;

  // Convert label in [A-Za-z] to subscript in [0, TOTAL_LABELS)
  auto get_subscript = [=](unsigned char label) -> uint8_t {
    return std::isupper(label) ? label - 'A' : label - 'a' + NUM_OF_LETTERS;
  };

  // Convert subscript in [0, 52) to label in [A-Za-z]
  auto get_label = [=](uint8_t s) -> unsigned char {
    return s < NUM_OF_LETTERS ? s + 'A' : s + 'a' - NUM_OF_LETTERS;
  };

  std::vector<uint64_t> label_count(TOTAL_LABELS, 0);
  std::vector<std::vector<uint8_t>> op_labels(1);
  std::vector<bool> op_has_ellipsis(1, false);

  // Parse equation for operand subscripts
  const auto arrow = equation.find("->");
  const auto lhs = equation.substr(0, arrow);
  for (auto it = lhs.begin(); it != lhs.end(); ++it) {
    switch (*it) {
      case ' ':
        // Ignore blank spaces
        break;
      case ',':
        // Begin processing next operand
        op_labels.emplace_back();
        op_has_ellipsis.push_back(false);
        break;
      case '.':
        // Parse ellipsis
        TORCH_CHECK(
            it + 2 != lhs.end() && *++it == '.' && *++it == '.',
            "einsum(): found \'.\' in equation that is not part of an ellipsis");
        TORCH_CHECK(
            !op_has_ellipsis.back(),
            "einsum(): at most one ellipsis can be specified per operand");
        op_labels.back().push_back(ELLIPSIS);
        op_has_ellipsis.back() = true;
        break;
      default:
        // Parse subscript
        TORCH_CHECK(
            std::isalpha(*it),
            "einsum(): expected subscripts to be in range [A-Za-z], but got ",
            *it);
        op_labels.back().push_back(get_subscript(*it));
        ++label_count[get_subscript(*it)];
        break;
    }
  }

  TORCH_CHECK(
      op_labels.size() == num_ops,
      "einsum(): the number of operands specified in the equation (",
      op_labels.size(),
      ") does not match the number of operands provided (",
      num_ops,
      ")");

  // Maximum number of dimensions covered by any ellipsis
  uint64_t ell_num_dim = 0;

  // Ensure all dimensions are covered for every operand
  for (const auto i : irange(num_ops)) {
    const uint64_t ndim = operands[i].ndimension();
    auto nlabel = op_labels[i].size();

    // Update maximum number of dimensions covered by ellipsis
    if (op_has_ellipsis[i]) {
      --nlabel;
      ell_num_dim = std::max(ell_num_dim, ndim - nlabel);
    }

    TORCH_CHECK(
        op_has_ellipsis[i] ? nlabel <= ndim : nlabel == ndim,
        "einsum(): the number of subscripts in the equation (",
        nlabel,
        op_has_ellipsis[i] ? ") is more than the number of dimensions ("
                           : ") does not match the number of dimensions (",
        ndim,
        ") for operand ",
        i,
        op_has_ellipsis[i] ? "" : " and no ellipsis was given");
  }

  // Indices in the permuted shape
  int64_t ell_index = -1;
  std::vector<int64_t> label_index(TOTAL_LABELS, -1);

  // Parse output equation
  uint64_t index = 0;
  if (arrow != string_view::npos) {
    const auto rhs = equation.substr(arrow + 2);
    for (auto it = rhs.begin(); it != rhs.end(); ++it) {
      switch (*it) {
        case ' ':
          break;
        case '.':
          TORCH_CHECK(
              it + 2 != rhs.end() && *++it == '.' && *++it == '.',
              "einsum(): found \'.\' in equation that is not part of an ellipsis");
          TORCH_CHECK(
              ell_index == -1,
              "einsum(): at most one ellipsis can be specified for the output");
          ell_index = index;
          index += ell_num_dim;
          break;
        default:
          // Parse subscript
          TORCH_CHECK(
              std::isalpha(*it),
              "einsum(): expected subscripts to be in range [A-Za-z], but got ",
              *it);
          TORCH_CHECK(
              label_index[get_subscript(*it)] == -1,
              "einsum(): subscript ",
              *it,
              " appears more than once for the output");
          TORCH_CHECK(
              label_count[get_subscript(*it)] > 0,
              "einsum(): output subscript",
              *it,
              " does not appear for any operand");
          label_index[get_subscript(*it)] = index++;
          break;
      }
    }
  } else {
    // Implicit output, ellipsis followed by labels with count == 1
    ell_index = index;
    index += ell_num_dim;
    for (const auto s : irange(TOTAL_LABELS)) {
      if (label_count[s] == 1) {
        label_index[s] = index++;
      }
    }
  }

  // Number of output dimensions
  const auto out_num_dim = index;

  // Assign index to ellipsis if not in the output
  if (ell_index == -1) {
    ell_index = index;
    index += ell_num_dim;
  }

  // Assign index to labels not in the output
  for (const auto s : irange(TOTAL_LABELS)) {
    if (label_count[s] > 0 && label_index[s] == -1) {
      label_index[s] = index++;
    }
  }

  // Next we check the sizes, take diagonals for repeated labels, unsqueeze
  // missing dimensions so all operands have the same dimensions and permute
  // the operands to align the dimensions following the indices compute above.
  // We also count how many operands have dimension with size > 1 for each
  // label used to identify which dimensions can be contracted.
  std::vector<int64_t> label_size(TOTAL_LABELS, 1);
  std::vector<int64_t> ell_sizes(ell_num_dim, 1);
  std::vector<uint64_t> dim_counts(index, 0);
  std::vector<Tensor> ops;
  for (const auto i : irange(num_ops)) {
    auto op = operands[i];
    std::vector<int64_t> permutation(index, -1);
    std::size_t dim = 0;
    for (const auto s : op_labels[i]) {
      if (s == ELLIPSIS) {
        // Iterate over each dimension covered by ellipsis
        const auto ndim = operands[i].ndimension() - (op_labels[i].size() - 1);
        for (auto j = ell_num_dim - ndim; j < ell_num_dim; ++j) {
          if (op.size(dim) != 1) {
            // Update ellipsis size
            TORCH_CHECK(
                ell_sizes[j] == 1 || ell_sizes[j] == op.size(dim),
                "einsum(): dimension ",
                dim,
                " covered by ellipsis in operand ",
                i,
                "has size ",
                op.size(dim),
                " which does not broadcast with previously seen ellipsis with size ",
                ell_sizes[j],
                " for the respective dimension");
            ell_sizes[j] = op.size(dim);
            ++dim_counts[ell_index + j];
          }
          permutation[ell_index + j] = dim++;
        }
      } else if (permutation[label_index[s]] == -1) {
        if (op.size(dim) != 1) {
          // Update subscript
          TORCH_CHECK(
              label_size[s] == 1 || label_size[s] == op.size(dim),
              "einsum(): subscript ",
              get_label(s),
              " has size ",
              op.size(dim),
              " for operand ",
              i,
              " which does not broadcast with previously seen size ",
              label_size[s]);
          label_size[s] = op.size(dim);
          ++dim_counts[label_index[s]];
        }
        permutation[label_index[s]] = dim++;
      } else {
        // Repeated label, take diagonal
        const auto prev_dim = permutation[label_index[s]];
        TORCH_CHECK(
            op.size(dim) == op.size(prev_dim),
            "einsum(): subscript ",
            get_label(s),
            " is repeated for operand ",
            i,
            " but the sizes don't match, ",
            op.size(dim),
            " != ",
            op.size(prev_dim));
        op = op.diagonal(0, prev_dim, dim).movedim(-1, prev_dim);
      }
    }
    // Add dimensions for missing labels
    for (auto& val : permutation) {
      if (val == -1) {
        op = op.unsqueeze(dim);
        val = dim++;
      }
    }
    ops.emplace_back(op.permute(permutation));
  }

  const auto path = optimize.value_or(std::vector<int64_t>{});
  auto it = path.begin();

  // Contract
  while (ops.size() > 1) {
    auto i = decltype(num_ops){0};
    auto j = decltype(num_ops){1};

    if (optimize.has_value()) {
      i = *it++;
      j = *it++;

      if (j < i) {
        std::swap(i, j);
      }

      TORCH_CHECK(
          i != j && i >= 0 && j < ops.size(),
          "einsum(): invalid contraction (",
          i,
          ", ",
          j,
          i == j ? ") cannot contract an operand with itself"
                 : ") operand index is out of bounds");
    }

    auto a = ops[i];
    auto b = ops[j];

    ops.erase(ops.begin() + j);
    ops.erase(ops.begin() + i);

    // Collect dimensions that can be summed now
    std::vector<int64_t> sum_dims;
    for (auto dim = out_num_dim; dim < index; ++dim) {
      if (a.size(dim) != 1 && b.size(dim) != 1) {
        if (--dim_counts[dim] == 1) {
          sum_dims.push_back(dim);
          dim_counts[dim] = 0;
        }
      } else if (dim_counts[dim] == 1) {
        if (a.size(dim) != 1) {
          a = a.sum(dim, true);
          dim_counts[dim] = 0;
        } else if (b.size(dim) != 1) {
          b = b.sum(dim, true);
          dim_counts[dim] = 0;
        }
      }
    }

    ops.emplace_back(sumproduct_pair(a, b, sum_dims, true));
  }

  // Sum out contraction dims
  if (index - out_num_dim > 0) {
    std::vector<int64_t> sum_dims(index - out_num_dim);
    std::iota(sum_dims.begin(), sum_dims.end(), out_num_dim);
    ops[0] = ops[0].sum(sum_dims);
  }

  return ops[0];
}

// _trilinear computes a trilinear einstein sum with an unrolled dimension
// the result is `(i1.unsqueeze(expand1)*i2.unsqueeze(expand2)*i2.unsqueeze(expand3)).sum(sumdim)`
// the computation is unrolled in the unroll_dim dimension
// its main purpose is to unify the computations in bilinear and bilinear_backward
Tensor _trilinear(const Tensor& i1_, const Tensor& i2_, const Tensor& i3_,
                  IntArrayRef expand1_, IntArrayRef expand2_, IntArrayRef expand3_,
                  IntArrayRef sumdim_, int64_t unroll_dim) {
  int64_t total_dim = i1_.dim()+expand1_.size();
  TORCH_CHECK((unroll_dim >= 0) && (unroll_dim < total_dim), "unroll_dim must be in [0,", total_dim-1, "]");
  auto expand1 = at::dim_list_to_bitset(expand1_, total_dim);
  auto expand2 = at::dim_list_to_bitset(expand2_, total_dim);
  auto expand3 = at::dim_list_to_bitset(expand3_, total_dim);
  auto sumdim  = at::dim_list_to_bitset(sumdim_,  total_dim);
  Tensor i1 = i1_;
  Tensor i2 = i2_;
  Tensor i3 = i3_;
  std::vector<int64_t> output_size;
  std::vector<int64_t> sum_dims_12, sum_dims_23;
  int64_t unroll_size = -1;
  // asserts...
  for (const auto i : c10::irange(total_dim)) {
    int64_t s = 0;
    if (expand1[i]) {
      i1 = i1.unsqueeze(i);
    } else  {
      s = i1.size(i);
    }
    if (expand2[i]) {
      i2 = i2.unsqueeze(i);
    } else  {
      s = i2.size(i);
    }
    if (expand3[i]) {
      i3 = i3.unsqueeze(i);
      if (sumdim[i] && (i != unroll_dim))
        sum_dims_12.push_back(i);
    } else  {
      s = i3.size(i);
      if (sumdim[i] && (i != unroll_dim))
        sum_dims_23.push_back(i);
    }
    output_size.push_back(sumdim[i] ? 1 : s);
    if (i == unroll_dim)
      unroll_size = s;
  }
  int64_t slicemul1 = (expand1[unroll_dim] ? 0 : 1);
  int64_t slicemul2 = (expand2[unroll_dim] ? 0 : 1);
  int64_t slicemul3 = (expand3[unroll_dim] ? 0 : 1);

  auto output = at::zeros(output_size, i1.options());

  // Three conditionals are necessary since this function is meant to work for both
  // forward and backward, which changes the dimensions of the inputs.
  // Note that if output has zero elems is because (at least) one of i1, i2, i3 has zero elems.
  if (i1.numel() != 0 && i2.numel() != 0 && i3.numel() != 0) {
    if (! sumdim[unroll_dim]) {
      for (const auto k : c10::irange(unroll_size)) {
        Tensor buf = at::native::sumproduct_pair(i1.narrow(unroll_dim, k * slicemul1, 1),
                                                 i2.narrow(unroll_dim, k * slicemul2, 1),
                                                 sum_dims_12, true);
        buf = at::native::sumproduct_pair(buf, i3.narrow(unroll_dim, k * slicemul3, 1), sum_dims_23, true);
        output.narrow(unroll_dim, k, 1).add_(buf);
      }
    }
    else {
      for (const auto k : c10::irange(unroll_size)) {
        Tensor buf = at::native::sumproduct_pair(i1.narrow(unroll_dim, k*slicemul1, 1),
                                                 i2.narrow(unroll_dim, k*slicemul2, 1), sum_dims_12, true);
        buf = at::native::sumproduct_pair(buf, i3.narrow(unroll_dim, k*slicemul3, 1), sum_dims_23, true);
        output.add_(buf);
      }
    }
  }
  for (int64_t i = output.dim()-1; i >= 0; i--)
    if (sumdim[i])
      output.squeeze_(i);
  return output;
}

Tensor bilinear(const Tensor& input1, const Tensor& input2, const Tensor& weight, const c10::optional<Tensor>& bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  TORCH_CHECK(input1.dim() == input2.dim(), "bilinear(): input dimensions do not match: got ", input1.dim(), " and ", input2.dim());
  for (const auto i : c10::irange(input1.dim() - 1)) {
    TORCH_CHECK(input1.size(i) == input2.size(i),
              "bilinear(): input batch dimensions do not match at dim ", i, ": got ", input1.size(i), " and ", input2.size(i));
  }
  TORCH_CHECK(input1.size(input1.dim() - 1) == weight.size(1),
            "bilinear(): input1 size does not match weight size: got ",
            input1.size(input1.dim() - 1), " but expected ", weight.size(1));
  TORCH_CHECK(input2.size(input2.dim() - 1) == weight.size(2),
            "bilinear(): input2 size does not match weight size: got ",
            input2.size(input2.dim() - 1), " but expected ", weight.size(2));
  TORCH_CHECK(!bias.defined() || bias.size(0) == weight.size(0),
            "bilinear(): bias size does not match weight size: got ",
            bias.size(0), " but expected ", weight.size(0));

  std::vector<int64_t> output_size;
  auto size1 = input1.sizes();
  output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
  output_size.push_back(weight.size(0));
  auto input1_flattened = input1.reshape({-1, input1.size(-1)});
  auto input2_flattened = input2.reshape({-1, input2.size(-1)});
  Tensor output = at::_trilinear(input1_flattened, weight, input2_flattened, {1,3}, {0}, {1,2}, {2,3}).reshape(output_size);
  if (bias.defined()) {
    output = output + bias;
  }
  return output;
}

// implements tensordot, a matrix-multiplication-like contraction, but the dimensions given
// in the two dimension lists
Tensor tensordot(const Tensor& input1, const Tensor& input2, IntArrayRef dims1, IntArrayRef dims2) {
  TORCH_CHECK(dims1.size() == dims2.size(), "both dimension lists should have same length");
  int64_t csize = 1;  // total size of the contracted dimensions
  Tensor t1 = input1;
  Tensor t2 = input2;
  for (const auto i : c10::irange(dims1.size())) {
    int s1 = input1.size(dims1[i]);
    int s2 = input2.size(dims2[i]);
    if (s2 == 1) { // broadcasted dimensions can be summed right away
      t1 = t1.sum(dims1[i], true);
    } else if (s1 == 1) {
      t2 = t2.sum(dims2[i], true);
    } else {
      TORCH_CHECK(s1 == s2, "contracted dimensions need to match, but first has size ", s1, " in dim ", dims1[i],
               " and second has size ", s2, " in dim ", dims2[i]);
      csize *= s1;
    }
  }

  auto cdims1 = at::dim_list_to_bitset(dims1, input1.dim());
  auto cdims2 = at::dim_list_to_bitset(dims2, input2.dim());
  std::vector<int64_t> p1, p2, rsizes;  // p1, p2: input permutations, rsizes: sizes of the result
  p1.reserve(input1.dim());
  p2.reserve(input2.dim());
  rsizes.reserve(input1.dim() + input2.dim() - (int64_t) dims1.size());
  int64_t size1 = 1; // number of non-contracted elements in input1
  int64_t size2 = 1; // number of non-contracted elements in input2

  // fill the permutations and compute sizes
  for (const auto i : c10::irange(input1.dim())) {
    if (! cdims1[i]) {
      p1.emplace_back(i);
      size1 *= t1.size(i);
      rsizes.emplace_back(t1.size(i));
    }
  }
  for (const auto x : dims1) {
    p1.emplace_back(x);
  }
  for (const auto x : dims2) {
    p2.emplace_back(x);
  }
  for (const auto i : c10::irange(input2.dim())) {
    if (! cdims2[i]) {
      p2.emplace_back(i);
      size2 *= t2.size(i);
      rsizes.emplace_back(t2.size(i));
    }
  }
  // permut and reshape for matrix multiplication
  t1 = t1.permute(p1).reshape({size1, csize});
  t2 = t2.permute(p2).reshape({csize, size2});
  // multiply and reshape to target size
  return at::mm(t1, t2).reshape(rsizes);
}

Tensor &tensordot_out(const Tensor& input1, const Tensor& input2, IntArrayRef dims1, IntArrayRef dims2, Tensor& result) {
  Tensor result_tmp = at::native::tensordot(input1, input2, dims1, dims2);
  auto result_dtype = result_tmp.scalar_type();
  auto output_tensor_dtype = result.scalar_type();
  auto output_device = result.device();
  auto input1_device = input1.device();
  auto input2_device = input2.device();
  // check if the input & output tensors are on the same device.
  TORCH_CHECK(
    (output_device == input1_device) && (input1_device == input2_device),
    "tensordot: Expected the output and input tensors to be on the "
    "same device, but got the output tensor on ", output_device,
    ", input tensor a on ", input1_device, ", and input tensor b on ", input2_device);
  // check if the computed result has the same dtype as the out tensor
  // (because tensordot does not support type promotion)
  TORCH_CHECK(
    result_dtype == output_tensor_dtype, "tensordot",
    ": Expected the output tensor to have dtype ", result_dtype,
    ", but got an output tensor with dtype ", output_tensor_dtype);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

}}  // namespace at::native
