#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/WrapDimUtilsMulti.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace at { namespace native {

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
  if (input.is_mkldnn()) {
    return at::mkldnn_linear(input, weight, bias);
  }
// #if defined(C10_MOBILE)
  if (xnnpack::use_linear(input, weight, bias)) {
    return xnnpack::linear(input, weight, bias);
  }
// #endif
  if (input.dim() == 2 && bias.defined()) {
    // Fused op is marginally faster.
    return at::addmm(bias, input, weight.t());
  }
  auto output = at::matmul(input, weight.t());
  if (bias.defined()) {
    output.add_(bias);
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
  for (int64_t i = 0; i < dim; i++) {
    auto sl = left.size(i)>1;
    auto sr = right.size(i)>1;
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

  for (auto it = lro.begin(); it != lro.end(); i++, it++) {
    opermutation[*it] = i;
  }
  for (auto it = lo.begin(); it != lo.end(); i++, it++) {
    opermutation[*it] = i;
  }
  for (auto it = sum_dims_.begin(); it != sum_dims_.end(); i++, it++) {
    opermutation[*it] = i;
  }
  for (auto it = ro.begin(); it != ro.end(); i++, it++) {
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
    for (int i = dim-1; i>=0; i--) {
      if (sum_dims[i]) {
        sizes.erase(sizes.begin() + i);
      }
    }
    result = result.view(sizes);
  }
  return result;
}

Tensor einsum(std::string eqn, TensorList tensors) {
  constexpr size_t number_of_letters = 26;
  std::string in_eqn;
  size_t pos;
  // The equation is given in terms of single lowercase letters ('a'..'z') and potentially an ellipsis.
  // Internally, we represent it using indices from 0 to num_total_dimensions, with each letter
  // mapped to an index and the ellipsis ('...') being mapped to a number of consequtive indices.
  // The mapping of letters to internal indices is given in letter_mapping. A value of -1 means that
  // the letter has not been assigned an index yet (because it has not been seen).
  // The ellipsis is defined by first_ell_idx (the first index) and num_ell_idxes (the number of indices).
  // A value of -1 for num_ell_idxes specifies that we have not seen an ellipsis yet.
  // Note: The internal indices are NOT the dimensions used internally. There is a mapping to them below.

  std::array<std::int64_t, number_of_letters> letter_mapping; // map letter to internal (numerical) label
  letter_mapping.fill(-1);
  int64_t num_ell_idxes = -1;
  int64_t first_ell_idx = 0;

  // The internal representation of the left hand side fo the equation (with ellipsis expanded) is stored in input_op_idxes.
  // For each operand, we have a vector mapping each dimension to an internal index.
  // We also keep track of the number of occurrences for each letter (to infer a right hand side if not given) and
  // of the last occurrence of each index.
  std::vector<std::vector<int64_t>> input_op_idxes;                   // the parsed operand indices
  std::array<std::int64_t, number_of_letters> num_letter_occurrences; // number of occurrence in the equation of this letter
  num_letter_occurrences.fill(0);
  std::vector<std::int64_t> last_idx_occurrence;                      // the last operator (left to right) using this index

  if ((pos = eqn.find("->")) != std::string::npos) { // check whether we have a right hand side. in_eq is the left hand side
    in_eqn = eqn.substr(0, pos);
  } else {
    in_eqn = eqn;
  }
  // remove spaces for einsum compatibility (#9929)
  in_eqn.erase(std::remove_if(in_eqn.begin(), in_eqn.end(), isspace), in_eqn.end());

  // next we parse in_eq (the left hand side) by iterating. It is a string of comma separated terms per index
  int64_t operand = 0;
  std::stringstream eqn_stream(in_eqn);
  std::string term;
  int64_t num_total_idxes = 0;
  while (! eqn_stream.eof()) {
    std::getline(eqn_stream, term, ',');  // term = string with indices of current term
    TORCH_CHECK((int64_t) tensors.size()>operand, "more operands in equation than tensors"); // we cannot have a longer equation than operands. We need to check here before we use the dimension

    int64_t ell_char_count = 0;            // handling of ellipsis '...' is a bit tedious, we count the '.'
    // if there is an ellipsis, the number of dimensions it represents must be total dim - letter dimensions
    int64_t candidate_num_ell_idxes = tensors[operand].dim() - term.size() + 3;
    int64_t dims_in_term = 0;              // dimensions we have seen
    std::vector<int64_t> current_op_idxes; // mapping of operand dimensions to indices for current term
    for (auto &c : term) {                 // c = character with a single letter or '.'
      if (c == '.') {
        ell_char_count++;
        TORCH_CHECK(ell_char_count <= 3, "can only have '.' in one ellispis '...' in term ", operand, " of the equation");
        if (ell_char_count == 3) {        // this completes the ellipsis
          if (num_ell_idxes == -1) {      // if we have not seen an ellipsis before, keep track of indices and size
            first_ell_idx = num_total_idxes;
            num_ell_idxes = candidate_num_ell_idxes;
            num_total_idxes += num_ell_idxes;
          }
          else {                          // we have seen an ellipsis before, so we check compatibility
            TORCH_CHECK(candidate_num_ell_idxes == num_ell_idxes,
                     "ellipsis must represent ", num_ell_idxes, " dimensions in all terms");
          }
          for (int64_t i = 0; i < num_ell_idxes; ++i) { // map ellipsis dimensions in operand to indices
            current_op_idxes.push_back(first_ell_idx + i);
            last_idx_occurrence.push_back(operand);
          }
          dims_in_term += num_ell_idxes;                // keep track of dimensions
        }
      } else {                                          // a letter (hopefully)
        TORCH_CHECK((ell_char_count == 0) || (ell_char_count == 3), "'.' must only occur in ellipsis, operand ", operand);
        TORCH_CHECK(('a' <= c) && (c <= 'z'), "only lowercase letters a-z allowed as indices");
        int64_t letter_num = c-'a';                     // letter_num  = position in letter_mapping
        if (letter_mapping[letter_num] == -1) {         // new letter, add internal index and mapping
          letter_mapping[letter_num] = num_total_idxes;
          num_total_idxes++;
          last_idx_occurrence.push_back(operand);
        } else {                                        // letter we have already seen
          last_idx_occurrence[letter_mapping[letter_num]] = operand;
        }
        num_letter_occurrences[letter_num]++;
        current_op_idxes.push_back(letter_mapping[letter_num]);
        dims_in_term++;
      }
    }
    TORCH_CHECK(dims_in_term == tensors[operand].dim(), "dimension mismatch for operand ", operand, ": equation ", dims_in_term, " tensor ", tensors[operand].dim());
    input_op_idxes.push_back(std::move(current_op_idxes));
    operand++;
  }
  // in the check below, we need ==, but > is captured above, so the error message can be specific that it is <.
  TORCH_CHECK((int64_t) tensors.size()==operand, "more tensors than operands in equation");

  // the following parses or infers output (right hand side)
  // it also assigns the idxes_to_preprocessed_dims (index -> dimension in preprocessed / output tensors)
  // for the output indices. -1 means that the index has not been assigned a dimension yet
  std::vector<int64_t> idxes_to_preprocessed_dims(num_total_idxes, -1);     // the position of the index in the tensor dimensions
  int64_t num_output_dims = 0;
  if (pos != std::string::npos) {            // parse the user provided right hand side
    int64_t ell_char_count = 0;
    for (auto &c : eqn.substr(pos+2)) {
      if (c == '.') {                        // '.' as part of ellipsis
        ell_char_count++;
        TORCH_CHECK(ell_char_count <= 3, "can only have '.' in one ellispis '...' in right hand side of the equation");
        if (ell_char_count == 3) {           // ellipsis complete
          TORCH_CHECK(num_ell_idxes >= 0, "ellipsis '...' may only appear in right hand side if it does in left hand side");
          for (int64_t i = 0; i < num_ell_idxes; ++i) {
            idxes_to_preprocessed_dims[first_ell_idx + i] = num_output_dims;
            num_output_dims++;
          }
        }
      } else if (! isspace(c)) {                              // letter (hopefully)
        TORCH_CHECK((ell_char_count == 0) || (ell_char_count == 3), "'.' must only occur in ellipsis in the right hand side");
        TORCH_CHECK(('a' <= c) && (c <= 'z'), "only lowercase letters a-z allowed as indices");
        int64_t letter_num = c-'a';
        TORCH_CHECK(idxes_to_preprocessed_dims[letter_mapping[letter_num]] == -1, "index ", c, " occurs twice in output");
        idxes_to_preprocessed_dims[letter_mapping[letter_num]] = num_output_dims;
        num_output_dims++;
      }
    }
  } else { // create an inferred right hand side
    // the ellipsis (if in the lhs) comes first
    if (num_ell_idxes >= 0) {
      for (int64_t i = 0; i < num_ell_idxes; ++i) {
        idxes_to_preprocessed_dims[first_ell_idx + i] = num_output_dims;
        num_output_dims++;
      }
    }
    // then the indices that occur exactly once in alphabetic order
    for (size_t idx = 0; idx < number_of_letters; idx++) {
      if (num_letter_occurrences[idx] == 1) {
        idxes_to_preprocessed_dims[letter_mapping[idx]] = num_output_dims;
        num_output_dims++;
      }
    }
  }
  // now we assign the idxes_to_preprocessed_dims (index -> dimension in preprocessed / output tensors)
  // for the non-output indices - those that are eventually summed over
  int64_t position = num_output_dims;
  for (int64_t i = 0; i < num_total_idxes; i++) {
    if (idxes_to_preprocessed_dims[i]==-1) {
      idxes_to_preprocessed_dims[i] = position;
      position++;
    }
  }

  // we now "homogenize the dimensions", i.e.
  // - take diagonals for duplicated indices
  // - permute the dimensions to match the order given by idxes_to_preprocessed_dims
  // - unsqueeze to create all dimensions for each index in each tensor where they are missing
  // we also check that sizes match
  // after this, all operands will have compatible shapes (i.e. all dimensions are aligned are broadcastable)
  std::vector<Tensor> preprocessed_operands;
  std::vector<std::int64_t> size_of_dims(num_total_idxes, -1); // keep track of sizes for each index, -1 means we have not seen a size yet
  for (int64_t op = 0; op < (int64_t) tensors.size(); op++) {
    auto preprocessed_op = tensors[op];
    std::vector<int64_t> idx_to_dim(num_total_idxes, -1); // the dimension which the index refers to in the original tensor, -1 means it does not appear
    std::vector<int64_t>& current_op_input_idxes = input_op_idxes[op];
    int64_t dim = 0; // there are two dimension indices: dim is after taking diagonals, i is in input
    for (size_t i = 0; i < current_op_input_idxes.size(); i++) {
      auto idx = current_op_input_idxes[i];
      auto dim_out = idxes_to_preprocessed_dims[idx];
      if (idx_to_dim[dim_out] == -1) { // first appearance
        idx_to_dim[dim_out] = dim;
        if (size_of_dims[idx] == -1) { // keep track of sizes
          size_of_dims[idx] = preprocessed_op.size(dim);
        }
        else {
          TORCH_CHECK(size_of_dims[idx] == preprocessed_op.size(dim), "size of dimension does not match previous size, operand ", op, ", dim ", i);
        }
        dim++;
      } else { // duplicate dimension in tensor --> take diagonal of idx_to_dim[dim_out] and dim and put the diagonal dimension to idx_to_dim[dim_out]
        TORCH_CHECK(size_of_dims[idx] == preprocessed_op.size(dim), "size of dimension does not match previous size, operand ", op, ", dim ", i);
        preprocessed_op = preprocessed_op.diagonal(0, idx_to_dim[dim_out], dim);
        // diagonal moves the diagonal dimension to the back
        // now we permute the last dim back to idx_to_dim[dim_out]
        std::vector<int64_t> perm(preprocessed_op.dim(), 0);
        for (int64_t d = 0; d < preprocessed_op.dim(); d++) {
          if (d == idx_to_dim[dim_out]) {
            perm[d] = preprocessed_op.dim() - 1;
          } else {
            perm[d] = d - (d > idx_to_dim[dim_out]);
          }
        }
        preprocessed_op = preprocessed_op.permute(perm);
      }
    }
    // now we permute the dimensions in the right order
    std::vector<int64_t> permutation; // permutation for this tensor
    for (auto &d : idx_to_dim) {
      if (d > -1) {
        permutation.push_back(d);
      }
    }
    preprocessed_op = preprocessed_op.permute(permutation);
    // finally, we insert dimensions for idxes not in the operand
    for (size_t dim = 0; dim < idx_to_dim.size(); dim++) {
      if (idx_to_dim[dim] == -1) {
        preprocessed_op = preprocessed_op.unsqueeze(dim);
      }
    }

    preprocessed_operands.push_back(std::move(preprocessed_op));
  }

  // now we reduce the indices from left to right
  // numpy allows to optimize the path using various
  // algorithms (see eigen_path in numpy docs)
  // we start with the leftmost operator and reduce indices that
  // appear only there
  Tensor result = std::move(preprocessed_operands[0]);
  for (int64_t idx = 0; idx < num_total_idxes; idx++) {
    if ((last_idx_occurrence[idx] == 0)
        && (idxes_to_preprocessed_dims[idx]>=num_output_dims)) {
      result = result.sum(idxes_to_preprocessed_dims[idx], true);
    }
  }

  // now we process each tensor using sumproduct_pair
  for (int64_t i = 1; i < (int64_t) preprocessed_operands.size(); i++) {
    std::vector<int64_t> sum_dims;
    for (int64_t idx = 0; idx < num_total_idxes; idx++) {
      if ((last_idx_occurrence[idx] == i)
          && (idxes_to_preprocessed_dims[idx]>=num_output_dims)) {
        sum_dims.push_back(idxes_to_preprocessed_dims[idx]);
      }
    }
    result = at::native::sumproduct_pair(result, std::move(preprocessed_operands[i]), sum_dims, true);
  }
  // finally, we squeeze out all non-result dimensions
  auto sizes = result.sizes().vec();
  for (int64_t dim = num_total_idxes-1; dim >= num_output_dims; dim--) {
    sizes.erase(sizes.begin() + dim);
  }

  result = result.view(sizes);
  return result;
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
  for (int64_t i = 0; i < total_dim; i++) {
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
  if (! sumdim[unroll_dim]) {
    for (int64_t k = 0; k < unroll_size; k++) {
      Tensor buf = at::native::sumproduct_pair(i1.narrow(unroll_dim, k * slicemul1, 1),
                                               i2.narrow(unroll_dim, k * slicemul2, 1),
                                               sum_dims_12, true);
      buf = at::native::sumproduct_pair(buf, i3.narrow(unroll_dim, k * slicemul3, 1), sum_dims_23, true);
      output.narrow(unroll_dim, k, 1).add_(buf);
    }
  }
  else {
    for (int64_t k = 0; k < unroll_size; k++) {
      Tensor buf = at::native::sumproduct_pair(i1.narrow(unroll_dim, k*slicemul1, 1),
                                               i2.narrow(unroll_dim, k*slicemul2, 1), sum_dims_12, true);
      buf = at::native::sumproduct_pair(buf, i3.narrow(unroll_dim, k*slicemul3, 1), sum_dims_23, true);
      output.add_(buf);
    }
  }
  for (int64_t i = output.dim()-1; i >= 0; i--)
    if (sumdim[i])
      output.squeeze_(i);
  return output;
}

Tensor bilinear(const Tensor& input1, const Tensor& input2, const Tensor& weight, const Tensor& bias) {
  TORCH_CHECK(input1.dim() == input2.dim(), "bilinear(): input dimensions do not match: got ", input1.dim(), " and ", input2.dim());
  for (int64_t i = 0; i < input1.dim() - 1; i++) {
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
  auto input1_flattened = input1.view({-1, input1.size(-1)});
  auto input2_flattened = input2.view({-1, input2.size(-1)});
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
  for (size_t i = 0; i < dims1.size(); i++) {
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
  for (int64_t i = 0; i < input1.dim(); i++) {
    if (! cdims1[i]) {
      p1.emplace_back(i);
      size1 *= t1.size(i);
      rsizes.emplace_back(t1.size(i));
    }
  }
  for (size_t i = 0; i < dims1.size(); i++) {
    p1.emplace_back(dims1[i]);
  }
  for (size_t i = 0; i < dims2.size(); i++) {
    p2.emplace_back(dims2[i]);
  }
  for (int64_t i = 0; i < input2.dim(); i++) {
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

}}  // namespace at::native
