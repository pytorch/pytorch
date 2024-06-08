#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/TensorOperators.h>
#include <c10/util/irange.h>
#include <c10/core/SymInt.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/TensorSubclassLikeUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_trilinear.h>
#include <ATen/ops/_trilinear_native.h>
#include <ATen/ops/add.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/bilinear_native.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/einsum_native.h>
#include <ATen/ops/linear_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mkldnn_linear.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/tensordot_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <cctype>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace at::native {

// Parse environment variable "TORCH_LINEAR_FLATTEN_3D"
static inline bool parseLinearFlatten3d() {
  // Uninitialized value
  static int value = -1;
  if (value == -1) {
    const char* env_str = std::getenv("TORCH_LINEAR_FLATTEN_3D");
    if (env_str != nullptr && strcmp(env_str, "1") == 0) {
      value = 1;
    } else {
      value = 0;
    }
  }
  return bool(value);
}

// `_flatten_nd_linear` flattens all but the last dimension of the input tensor
// before passing it to linear operation
static inline Tensor _flatten_nd_linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    const auto input_sizes = input.sym_sizes();
    // can't use -1 in reshape because it errors when a dimension is 0
    c10::SymInt flattened_dim = 1;
    for (int64_t i = 0, ndim = input_sizes.size(); i < ndim - 1; ++i) {
      flattened_dim = flattened_dim * input_sizes[i];
    }
    auto inp_reshape = input.reshape_symint({flattened_dim, input_sizes.at(input_sizes.size() -1)});
    const auto result = at::addmm(bias, inp_reshape, weight.t());
    auto new_size = input_sizes.slice(0, input_sizes.size() - 1);
    c10::SymDimVector sizes_vec(new_size.begin(), new_size.end());
    sizes_vec.push_back(result.sym_size(1));
    return result.view_symint(sizes_vec);
}


Tensor linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt) {
  // _matmul_impl checks this again later, but _flatten_nd_linear does not work on scalars inputs,
  // so let's try to catch this here already
  const auto input_dim = input.dim();
  const auto weight_dim = weight.dim();
  TORCH_CHECK(input_dim != 0 && weight_dim != 0,
              "both arguments to linear need to be at least 1D, but they are ",
              input_dim, "D and ", weight_dim, "D");

  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
    : c10::MaybeOwned<Tensor>::owned(std::in_place);
  if (input.is_mkldnn()) {
    return at::mkldnn_linear(input, weight, *bias);
  }
#if defined(C10_MOBILE)
  if (xnnpack::use_linear(input, weight, *bias)) {
    return xnnpack::linear(input, weight, *bias);
  }
#endif
  if (input_dim == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::addmm(*bias, input, weight.t());
  }
  if (bias->defined() && !input.is_xla()) {
    // Also hit the fused path for contiguous 3D input, if not using xla
    // backend. Reshaping/flattening has some performance implications on xla.
    if (input.is_contiguous() && input_dim == 3) {
      return _flatten_nd_linear(input, weight, *bias);
    } else if (input.is_contiguous() && input.layout() == c10::kStrided && weight.layout() == c10::kStrided && bias->dim() == 1) {
      return _flatten_nd_linear(input, weight, *bias);
    } else if (parseLinearFlatten3d() && input_dim == 3) {
      // If user forces flattening via env var
      const Tensor input_cont = input.contiguous();
      return _flatten_nd_linear(input_cont, weight, *bias);
    }
  }
  auto output = at::matmul(input, weight.t());
  if (bias->defined()) {
    // for composite compliance use out-of-place version of `add`
    if (isTensorSubclassLike(*bias) ||
        bias->_fw_grad(/*level*/ 0).defined()) {
      output = at::add(output, *bias);
    } else {
      output.add_(*bias);
    }
  }
  return output;
}

Tensor& linear_out(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt, Tensor& output) {
  TORCH_CHECK(!input.is_mkldnn(), "linear doesn't support out for MKLDNN tensors");
  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
              ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
              : c10::MaybeOwned<Tensor>::owned(std::in_place);

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
  if (sum_dims_.empty())
    return at::mul(left_, right_);
  int64_t dim = left_.dim();
  auto sum_dims = at::dim_list_to_bitset(sum_dims_, dim);
  // dimensions that will be part of the output (i.e. not summed over) in three vectors:
  // dims in lro appear in left, right and output, similarly, lo: left and output, ro: right and output
  // also the sizes are kept track of for reshaping
  std::vector<int64_t> lro, lo, ro;
  SymInt lro_size = 1, lo_size = 1, ro_size = 1, sum_size = 1;
  Tensor left = left_;
  Tensor right = right_;
  for (const auto i : c10::irange(dim)) {
    auto sl = left.sym_size(i)!=1;
    auto sr = right.sym_size(i)!=1;
    if (sum_dims[i]) { // first dimensions that will be summed over after multiplication
      if (sl && sr) {  // dimensions nontrivially in both left and right must be of the same size
        TORCH_CHECK(left.sym_size(i)==right.sym_size(i), "non-broadcast dimensions must match");
        sum_size *= left.sym_size(i);
      } else if (sl) { // if it is only in one of left and right, we can sum right away
        left = left.sum(i, true);
      } else if (sr) {
        right = right.sum(i, true);
      }
    } else if (sl && sr) { // now deal with dimensions that will be in the output
      // dimensions nontrivially in both left and right must be of the same size
      TORCH_CHECK(left.sym_size(i)==right.sym_size(i), "non-broadcast dimensions must match");
      lro.push_back(i);
      lro_size *= left.sym_size(i);
    } else if (sl) { // keep track of dimensions appearing only once
      lo.push_back(i);
      lo_size *= left.sym_size(i);
    } else {
      ro.push_back(i);
      ro_size *= right.sym_size(i);
    }
  }
  // we now work with the following permutations / shapes.
  // the pipeline is permute inputs -> reshape inputs -> batch matrix mul -> reshape(view) output -> permute output
  // output: "lro, lo, 1-for-summed-dims, ro" with original shape dimensions
  // left:   "lro, lo, summed" permuted with lpermutation and the three flattened
  // right:  "lro, summed, ro" permuted with rpermutation and the three flattened
  // then the permuted output is a view of bmm(left, right)
  // finally, opermutation reverts the permutation to the original order of dimensions
  auto out_num_dim = lro.size() + lo.size() + sum_dims_.size() + ro.size();
  std::vector<SymInt> out_size;
  out_size.reserve(out_num_dim);
  for (auto& d : lro) out_size.push_back(left.sym_size(d));
  for (auto& d : lo) out_size.push_back(left.sym_size(d));
  for (auto& d : sum_dims_) { out_size.emplace_back(1); (void)(d); }; // avoid warning about not using d
  for (auto& d : ro) out_size.push_back(right.sym_size(d));

  std::vector<int64_t> lpermutation(lro);
  lpermutation.insert(lpermutation.end(), lo.begin(), lo.end());
  lpermutation.insert(lpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  lpermutation.insert(lpermutation.end(), ro.begin(), ro.end());

  std::vector<int64_t> rpermutation(lro);
  rpermutation.insert(rpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  rpermutation.insert(rpermutation.end(), ro.begin(), ro.end());
  rpermutation.insert(rpermutation.end(), lo.begin(), lo.end());

  std::vector<int64_t> opermutation(out_num_dim, -1);
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
  left = left.permute(lpermutation).reshape_symint({lro_size, std::move(lo_size), sum_size});
  right = right.permute(rpermutation).reshape_symint({std::move(lro_size), std::move(sum_size), std::move(ro_size)});
  Tensor result = at::bmm(left, right);
  result = result.view_symint(out_size).permute(opermutation);

  // finally squeeze summed dimensions if desired
  if (! keepdim) {
    auto sizes = result.sizes().vec();
    for (auto i = dim-1; i>=0; i--) {
      if (sum_dims[i]) {
        sizes.erase(sizes.begin() + i);
      }
    }
    result = result.view(sizes);
  }
  return result;
}

// There are roughly three parts to computing einsum:
// 1. Parse equation to extract the labels for each input operand and output
// 2. Unsqueeze missing dimensions from input operands and permute to align them
// 3. Compute result by multiplying input operands and summing contraction
//    dimensions. We do the last part by reducing to bmm.
// If a path is specified, we reduce in the order specified by the path, else we
// default to going left => right. The path is a list of indices processed the same
// way as opt-einsum: https://optimized-einsum.readthedocs.io/en/stable/path_finding.html#format-of-the-path
Tensor einsum(c10::string_view equation, TensorList operands, at::OptionalIntArrayRef path) {
  TORCH_CHECK(!operands.empty(), "einsum(): must provide at least one operand");
  const auto num_ops = operands.size();

  if (path.has_value()) {
    const auto path_size = num_ops == 1 ? 1 : (num_ops - 1) * 2;
    TORCH_CHECK(
        path->size() == path_size,
        "einsum(): expected contraction path given in path parameter to have size ",
        path_size,
        " but got ",
        path->size());
  }

  // Labels must be in range [A-Za-z]
  constexpr uint8_t NUM_OF_LETTERS = 'z' - 'a' + 1;
  constexpr uint8_t TOTAL_LABELS = NUM_OF_LETTERS * 2;

  // Code used to identify ELLIPSIS ("...")
  constexpr uint8_t ELLIPSIS = TOTAL_LABELS;

  // Convert label in [A-Za-z] to subscript in [0, TOTAL_LABELS)
  auto label_to_subscript = [=](unsigned char label) -> uint8_t {
    return std::isupper(label) ? label - 'A' : label - 'a' + NUM_OF_LETTERS;
  };

  // Convert subscript in [0, TOTAL_LABELS) to label in [A-Za-z]
  auto subscript_to_label = [=](uint8_t s) -> unsigned char {
    return s < NUM_OF_LETTERS ? s + 'A' : s + 'a' - NUM_OF_LETTERS;
  };

  // Find arrow (->) to split equation into lhs and rhs
  const auto arrow_pos = equation.find("->");
  const auto lhs = equation.substr(0, arrow_pos);

  // Convert labels for input operands into an index in [0, 52) and store
  // them in op_labels for each operand along with ELLIPSIS if present.
  std::vector<std::vector<uint8_t>> op_labels(num_ops);
  bool ell_in_input = false;
  std::size_t curr_op = 0;
  for (std::size_t i = 0; i < lhs.length(); ++i) {
    const unsigned char label = lhs[i];
    switch (label) {
      case ' ':
        // Ignore spaces
        break;

      case '.':
        TORCH_CHECK(
            // Only one ellipsis per operand can be given
            !ell_in_input,
            "einsum(): found \'.\' for operand ",
            curr_op,
            " for which an ellipsis was already found");
        TORCH_CHECK(
            // Ensure it's a valid ellipsis
            i + 2 < lhs.length() && lhs[++i] == '.' && lhs[++i] == '.',
            "einsum(): found \'.\' for operand ",
            curr_op,
            " that is not part of any ellipsis");
        op_labels[curr_op].push_back(ELLIPSIS);
        ell_in_input = true;
        break;

      case ',':
        // Move onto next operand
        ++curr_op;
        TORCH_CHECK(
            curr_op < num_ops,
            "einsum(): fewer operands were provided than specified in the equation");
        ell_in_input = false;
        break;

      default:
        // Parse label
        TORCH_CHECK(
            std::isalpha(label),
            "einsum(): invalid subscript given at index ",
            i,
            " in the equation string, subscripts must be in [a-zA-Z]");
        op_labels[curr_op].push_back(label_to_subscript(label));
    }
  }

  TORCH_CHECK(
      curr_op == num_ops - 1,
      "einsum(): more operands were provided than specified in the equation");

  std::vector<int64_t> label_count(TOTAL_LABELS, 0);

  // The maximum number of dimensions covered by any ellipsis, needed when
  // unsqueezing missing dimensions from operands to permute and broadcast
  int64_t ell_num_dim = 0;

  // Compute label frequency and number of dimensions covered by ellipsis
  // We do this after parsing labels to make it more readable and simpler
  // to compute the number of dimensions covered by ellipsis.
  for(const auto i : c10::irange(num_ops)) {
    const auto& operand = operands[i];
    const auto labels = op_labels[i];
    const auto ndims = operand.dim();
    int64_t nlabels = static_cast<int64_t>(labels.size());
    bool has_ellipsis = false;

    for (const auto& label : labels) {
      if (label == ELLIPSIS) {
        --nlabels;
        has_ellipsis = true;
        ell_num_dim = std::max(ell_num_dim, ndims - nlabels);
      } else {
        ++label_count[label];
      }
    }

    TORCH_CHECK(
        has_ellipsis ? nlabels <= ndims : nlabels == ndims,
        "einsum(): the number of subscripts in the equation (",
        nlabels,
        has_ellipsis ? ") is more than the number of dimensions ("
                     : ") does not match the number of dimensions (",
        ndims,
        ") for operand ",
        i,
        has_ellipsis ? "" : " and no ellipsis was given");
  }

  // We want to align the dimensions of every input tensor to have
  // shape out_dims + sum_dims. For this, we create a mapping of label
  // to index into the permuted shape.
  std::vector<int64_t> label_perm_index(TOTAL_LABELS, -1);

  // Current index in the permuted shape
  int64_t perm_index = 0;

  // Start index of ellipsis dimensions in the permuted shape
  int64_t ell_index = 0;
  bool ell_in_output = false;

  if (arrow_pos == std::string::npos) {
    // Implicit output is ellipsis (...) + labels seen only once
    perm_index = ell_num_dim;
    // ell_in_output is used to stop us from reducing ellipses dims later
    ell_in_output = true;
    for (const auto label : c10::irange(TOTAL_LABELS)) {
      if (label_count[label] == 1) {
        label_perm_index[label] = perm_index++;
      }
    }
  } else {
    // Parse explicit output
    const auto rhs = equation.substr(arrow_pos + 2);
    for (std::size_t i = 0; i < rhs.length(); ++i) {
      const unsigned char label = rhs[i];
      switch (label) {
        case ' ':
          // Ignore spaces
          break;

        case '.':
          TORCH_CHECK(
              // There can only be one ellipsis in the output
              !ell_in_output,
              "einsum(): found \'.\' for output but an ellipsis (...) was already found");
          TORCH_CHECK(
              // Ensure ellipsis is correct
              i + 2 < rhs.length() && rhs[++i] == '.' && rhs[++i] == '.',
              "einsum(): found \'.\' for output that is not part of any ellipsis (...)");
          ell_index = perm_index;
          perm_index += ell_num_dim;
          ell_in_output = true;
          break;

        default:
          TORCH_CHECK(
              std::isalpha(label),
              "einsum(): invalid subscript given at index ",
              lhs.size() + 2 + i,
              " in the equation string, subscripts must be in [a-zA-Z]");
          const auto index = label_to_subscript(label);
          TORCH_CHECK(
              // Ensure label appeared at least once for some input operand and at
              // most once for the output
              label_count[index] > 0 && label_perm_index[index] == -1,
              "einsum(): output subscript ",
              label,
              label_perm_index[index] > -1
                  ? " appears more than once in the output"
                  : " does not appear in the equation for any input operand");
          label_perm_index[index] = perm_index++;
      }
    }
  }

  // Save number of dimensions in output before adding contraction dims (dims to sum out)
  const int64_t out_num_dim = perm_index;

  // If ellipsis is not part of the output, add to contraction dimensions
  if (!ell_in_output) {
    ell_index = perm_index;
    perm_index += ell_num_dim;
  }

  // Add contraction labels (labels not present in output)
  for (const auto label : c10::irange(TOTAL_LABELS)) {
    if (label_count[label] > 0 && label_perm_index[label] == -1) {
      label_perm_index[label] = perm_index++;
    }
  }

  // Next: we check the sizes, take diagonals for repeated labels, unsqueeze
  // missing dimensions so all operands have the same dimensions and permute
  // the operands to align the dimensions following the indices computed above.
  // We also count how many operands have dimension with size != 1 for each
  // label used to identify which dimensions can be contracted.
  std::vector<SymInt> label_size(TOTAL_LABELS, 1);
  std::vector<SymInt> ell_sizes(ell_num_dim, 1);
  std::vector<uint64_t> dim_counts(perm_index, 0);
  std::deque<Tensor> ops;
  for (const auto i : irange(num_ops)) {
    auto op = operands[i];
    std::vector<int64_t> permutation(perm_index, -1);
    std::int64_t dim = 0;
    for (const auto s : op_labels[i]) {
      if (s == ELLIPSIS) {
        // Iterate over each dimension covered by ellipsis
        const auto ndim = operands[i].ndimension() - (static_cast<int64_t>(op_labels[i].size()) - 1);
        for (auto j = ell_num_dim - ndim; j < ell_num_dim; ++j) {
          if (op.sym_size(dim) != 1) {
            // Update ellipsis size
            TORCH_CHECK(
                ell_sizes[j] == 1 || ell_sizes[j] == op.sym_size(dim),
                "einsum(): dimension ",
                dim,
                " covered by ellipsis in operand ",
                i,
                "has size ",
                op.size(dim),
                " which does not broadcast with previously seen ellipsis with size ",
                ell_sizes[j],
                " for the respective dimension");
            ell_sizes[j] = op.sym_size(dim);
            ++dim_counts[ell_index + j];
          }
          permutation[ell_index + j] = dim++;
        }
      } else if (permutation[label_perm_index[s]] == -1) {
        if (op.sym_size(dim) != 1) {
          // Update subscript
          TORCH_CHECK(
              label_size[s] == 1 || label_size[s] == op.sym_size(dim),
              "einsum(): subscript ",
              subscript_to_label(s),
              " has size ",
              op.sym_size(dim),
              " for operand ",
              i,
              " which does not broadcast with previously seen size ",
              label_size[s]);
          label_size[s] = op.sym_size(dim);
          ++dim_counts[label_perm_index[s]];
        }
        permutation[label_perm_index[s]] = dim++;
      } else {
        // Repeated label, take diagonal
        const auto prev_dim = permutation[label_perm_index[s]];
        TORCH_CHECK(
          op.sym_size(dim) == op.sym_size(prev_dim),
            "einsum(): subscript ",
            subscript_to_label(s),
            " is repeated for operand ",
            i,
            " but the sizes don't match, ",
            op.sym_size(dim),
            " != ",
            op.sym_size(prev_dim));
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

  const auto contract_path = path.value_or(std::vector<int64_t>{});
  auto it = contract_path.begin();

  // Contract
  while (ops.size() > 1) {
    int64_t i = 0;
    int64_t j = 1;

    if (path.has_value()) {
      i = *it++;
      j = *it++;
      if (j < i) {
        std::swap(i, j);
      }

      TORCH_CHECK(
          i != j && i >= 0 && j < static_cast<int64_t>(ops.size()),
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
    SmallVector<int64_t, 5> a_dims_to_sum;
    SmallVector<int64_t, 5> b_dims_to_sum;
    for (auto dim = out_num_dim; dim < perm_index; ++dim) {
      if (a.sym_size(dim) != 1 && b.sym_size(dim) != 1) {
        if (--dim_counts[dim] == 1) {
          sum_dims.push_back(dim);
          dim_counts[dim] = 0;
        }
      } else if (dim_counts[dim] == 1) {
        if (a.sym_size(dim) != 1) {
          a_dims_to_sum.push_back(dim);
          dim_counts[dim] = 0;
        } else if (b.sym_size(dim) != 1) {
          b_dims_to_sum.push_back(dim);
          dim_counts[dim] = 0;
        }
      }
    }

    // Sum multiple dims at a time to minimize the number of kernel calls to sum
    if (!a_dims_to_sum.empty()) {
      a = a.sum(a_dims_to_sum, true);
    }
    if (!b_dims_to_sum.empty()) {
      b = b.sum(b_dims_to_sum, true);
    }

    if (path.has_value()) {
      ops.emplace_back(sumproduct_pair(a, b, sum_dims, true));
    } else {
      ops.emplace_front(sumproduct_pair(a, b, sum_dims, true));
    }
  }

  // Sum out contraction dims
  if (perm_index - out_num_dim > 0) {
    // if there were ops to contract, we would have already done so
    // in the previous loop and all the dims to sum are now 1
    // NB: use view instead of squeeze (or sum) for faster (mps) performance
    if (num_ops > 1) {
      auto sizes = ops[0].sym_sizes().vec();
      for (auto dim = perm_index - 1; dim >= out_num_dim; --dim) {
        sizes.erase(sizes.begin() + dim);
      }
      return ops[0].view_symint(sizes);
    } else {
      std::vector<int64_t> sum_dims(perm_index - out_num_dim);
      std::iota(sum_dims.begin(), sum_dims.end(), out_num_dim);
      return ops[0].sum(sum_dims);
    }
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
  std::vector<c10::SymInt> output_size;
  std::vector<int64_t> sum_dims_12, sum_dims_23;
  int64_t unroll_size = -1;
  // asserts...
  for (const auto i : c10::irange(total_dim)) {
    c10::SymInt s = 0;
    if (expand1[i]) {
      i1 = i1.unsqueeze(i);
    } else  {
      s = i1.sym_size(i);
    }
    if (expand2[i]) {
      i2 = i2.unsqueeze(i);
    } else  {
      s = i2.sym_size(i);
    }
    if (expand3[i]) {
      i3 = i3.unsqueeze(i);
      if (sumdim[i] && (i != unroll_dim))
        sum_dims_12.push_back(i);
    } else  {
      s = i3.sym_size(i);
      if (sumdim[i] && (i != unroll_dim))
        sum_dims_23.push_back(i);
    }
    output_size.push_back(sumdim[i] ? 1 : s);
    if (i == unroll_dim)
      unroll_size = s.guard_int(__FILE__, __LINE__);
  }
  int64_t slicemul1 = (expand1[unroll_dim] ? 0 : 1);
  int64_t slicemul2 = (expand2[unroll_dim] ? 0 : 1);
  int64_t slicemul3 = (expand3[unroll_dim] ? 0 : 1);

  auto output = at::zeros_symint(output_size, i1.options());

  // Three conditionals are necessary since this function is meant to work for both
  // forward and backward, which changes the dimensions of the inputs.
  // Note that if output has zero elems is because (at least) one of i1, i2, i3 has zero elems.
  if (i1.sym_numel() != 0 && i2.sym_numel() != 0 && i3.sym_numel() != 0) {
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

Tensor bilinear(const Tensor& input1, const Tensor& input2, const Tensor& weight, const std::optional<Tensor>& bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  if (bias.defined()) {
    TORCH_CHECK(
        input1.dtype() == input2.dtype() && input1.dtype() == weight.dtype() &&
            input1.dtype() == bias.dtype(),
        "All tensors must have the same dtype, got input1: ",
        input1.dtype(),
        ", input2: ",
        input2.dtype(),
        ", weight: ",
        weight.dtype(),
        ", bias: ",
        bias.dtype());
  } else {
    TORCH_CHECK(
        input1.dtype() == input2.dtype() && input1.dtype() == weight.dtype(),
        "All tensors must have the same dtype, got input1: ",
        input1.dtype(),
        ", input2: ",
        input2.dtype(),
        ", weight: ",
        weight.dtype());
  }

  TORCH_CHECK(input1.dim() == input2.dim(), "bilinear(): input dimensions do not match: got ", input1.dim(), " and ", input2.dim());
  for (const auto i : c10::irange(input1.dim() - 1)) {
    TORCH_CHECK(input1.sym_size(i) == input2.sym_size(i),
              "bilinear(): input batch dimensions do not match at dim ", i, ": got ", input1.sym_size(i), " and ", input2.sym_size(i));
  }
  TORCH_CHECK(input1.sym_size(input1.dim() - 1) == weight.sym_size(1),
            "bilinear(): input1 size does not match weight size: got ",
            input1.sym_size(input1.dim() - 1), " but expected ", weight.sym_size(1));
  TORCH_CHECK(input2.sym_size(input2.dim() - 1) == weight.sym_size(2),
            "bilinear(): input2 size does not match weight size: got ",
            input2.sym_size(input2.dim() - 1), " but expected ", weight.sym_size(2));
  TORCH_CHECK(!bias.defined() || bias.sym_size(0) == weight.sym_size(0),
            "bilinear(): bias size does not match weight size: got ",
            bias.sym_size(0), " but expected ", weight.sym_size(0));

  std::vector<c10::SymInt> output_size;
  auto size1 = input1.sym_sizes();
  output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
  output_size.push_back(weight.sym_size(0));
  auto input1_flattened = input1.reshape_symint({-1, input1.sym_size(-1)});
  auto input2_flattened = input2.reshape_symint({-1, input2.sym_size(-1)});
  Tensor output = at::_trilinear(input1_flattened, weight, input2_flattened, {1,3}, {0}, {1,2}, {2,3}).reshape_symint(output_size);
  if (bias.defined()) {
    output = output + bias;
  }
  return output;
}

// implements tensordot, a matrix-multiplication-like contraction, but the dimensions given
// in the two dimension lists
Tensor tensordot(const Tensor& input1, const Tensor& input2, IntArrayRef dims1, IntArrayRef dims2) {
  TORCH_CHECK(dims1.size() == dims2.size(), "both dimension lists should have same length");
  TORCH_CHECK(input1.scalar_type() == input2.scalar_type(), "both inputs should have same dtype");
  SymInt csize = 1;  // total size of the contracted dimensions
  Tensor t1 = input1;
  Tensor t2 = input2;
  for (const auto i : c10::irange(dims1.size())) {
    SymInt s1 = input1.sym_size(dims1[i]);
    SymInt s2 = input2.sym_size(dims2[i]);
    if (s2 == 1) { // broadcasted dimensions can be summed right away
      t1 = t1.sum(dims1[i], true, t1.scalar_type());
    } else if (s1 == 1) {
      t2 = t2.sum(dims2[i], true, t2.scalar_type());
    } else {
      TORCH_CHECK(s1 == s2, "contracted dimensions need to match, but first has size ", s1, " in dim ", dims1[i],
               " and second has size ", s2, " in dim ", dims2[i]);
      csize *= s1;
    }
  }

  auto cdims1 = at::dim_list_to_bitset(dims1, input1.dim());
  auto cdims2 = at::dim_list_to_bitset(dims2, input2.dim());
  std::vector<int64_t> p1, p2;  // p1, p2: input permutations
  std::vector<SymInt> rsizes;  // rsizes: sizes of the result
  p1.reserve(input1.dim());
  p2.reserve(input2.dim());
  rsizes.reserve(input1.dim() + input2.dim() - (int64_t) dims1.size());
  SymInt size1 = 1; // number of non-contracted elements in input1
  SymInt size2 = 1; // number of non-contracted elements in input2

  // fill the permutations and compute sizes
  for (const auto i : c10::irange(input1.dim())) {
    if (! cdims1[i]) {
      p1.emplace_back(i);
      size1 *= t1.sym_size(i);
      rsizes.emplace_back(t1.sym_size(i));
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
      size2 *= t2.sym_size(i);
      rsizes.emplace_back(t2.sym_size(i));
    }
  }
  // permute and reshape for matrix multiplication
  t1 = t1.permute(p1).reshape_symint({size1, csize});
  t2 = t2.permute(p2).reshape_symint({csize, size2});
  // multiply and reshape to target size
  return at::mm(t1, t2).reshape_symint(rsizes);
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

}  // namespace at::native
