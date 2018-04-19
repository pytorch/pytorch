#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtilsMulti.h"


namespace at { namespace native {


// sumproduct_pair computes `(left*right).sum(sumdims)` by means of permutation and
// batch matrix multiplication
// its main purpose is to provide a pairwise reduction for einsum
static Tensor sumproduct_pair(const Tensor& left_, const Tensor& right_, IntList sum_dims_, bool keepdim) {
  // assumes that tensors have been pre-unsqueezed (so that all dimensions match - after broadcasting)
  // but makes no other assumptions on the order of dimensions
  AT_ASSERT(left_.dim()==right_.dim(), "number of dimensions must match");
  if (sum_dims_.size() == 0)
    return at::mul(left_, right_);
  int64_t dim = left_.dim();
  auto sum_dims = dim_list_to_bitset(sum_dims_, dim);
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
	AT_ASSERT(left.size(i)==right.size(i), "non-broadcast dimensions must match");
	sum_size *= left.size(i);
      } else if (sl) { // if it is only in one of left and right, we can sum right away
	left = left.sum(i, true);
      } else if (sr) {
	right = right.sum(i, true);
      }
    } else if (sl && sr) { // now deal with dimensions  dimensions that will be in the output
      // dimensions nontrivially in both left and right must be of the same size
      AT_ASSERT(left.size(i)==right.size(i), "non-broadcast dimensions must match");
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
    for (int i = dim-1; i>=0; i--)
      if (sum_dims[i])
	result.squeeze_(i);
  }
  return result;
}

Tensor einsum(std::string eqn, TensorList tensors) {
  constexpr size_t number_of_letters = 26;
  std::string in_eqn;
  size_t pos;
  // we need are number of mappings (letter) index for analysing the equation. The index runs from 0='a' through 25='z'.
  std::array<std::int64_t, number_of_letters> number_of_occurrences; // number of occurrence in the equation of this index
  number_of_occurrences.fill(0);
  std::array<std::int64_t, number_of_letters> last_occurrence;      // the last operator (left to right) using this index
  last_occurrence.fill(-1);

  if ((pos = eqn.find("->")) != std::string::npos) { // check whether we have a right hand side. in_eq is the left hand side
    in_eqn = eqn.substr(0, pos);
  } else {
    in_eqn = eqn;
  }

  // next we parse in_eq (the left hand side) by iterating. It is a string of comma separated terms per index
  int64_t operand = 0;
  std::stringstream eqn_stream(in_eqn);
  std::string term;
  while (! eqn_stream.eof()) {
    std::getline(eqn_stream, term, ',');  // term = string with indices of current term
    int64_t dims_in_operand = 0;
    for (auto &c : term) {                // c    = character with a single index
      AT_ASSERT(('a' <= c) && (c <= 'z'), "only lowercase letters a-z allowed as indices");
      int64_t index_num = c-'a';          // index_num  = index to be used in the vectors above
      number_of_occurrences[index_num]++;
      // when there are two occurrences we need to take a diagonal with respect to the dimensions
      // occuring multiple times before continuing the processing.
      // e.g. einsum('ii->i', [A]) should return the diagonal
      // This waits for the general diagonal handling discussed in #6479
      // for now, we error out here
      AT_ASSERT(last_occurrence[index_num] < operand, "diagonals (multiple occurrences of the same index for one tensor) not implemented yet")
      last_occurrence[index_num] = operand;
      dims_in_operand++;
    }
    AT_ASSERT((int64_t) tensors.size()>operand, "more operands in equation than tensors"); // we cannot have a longer equation than operands. We need to check here before we check the dimensions
    AT_ASSERT(dims_in_operand == tensors[operand].dim(), "dimension mismatch for operand %zd: equation %zd, tensor %zd", operand, dims_in_operand, tensors[operand].dim());
    operand++;
  }
  AT_ASSERT((int64_t) tensors.size()==operand, "more tensors than operands in equation");  // we need ==, but > is captured above, so the error message can be specific that it is <.

  // the following parses or infers output (right hand side)
  // it also assigns the sorted_positions ((letter) index -> dimension in Tensors) and position_labels (dimensions in Tensors -> index)
  // for the output indices
  std::array<std::int64_t, number_of_letters> sorted_position;     // the position of the index in the tensor dimensions
  sorted_position.fill(-1);
  int64_t num_output_dims = 0;
  std::vector<int64_t> position_labels;
  if (pos != std::string::npos) {            // parse the user provided right hand side
    for (auto &c : eqn.substr(pos+2)) {
      AT_ASSERT(('a' <= c) && (c <= 'z'), "only lowercase letters a-z allowed as indices");
      int64_t index_num = c-'a';
      AT_ASSERT(sorted_position[index_num] == -1, "index %c occurs twice in output", c);
      sorted_position[index_num] = num_output_dims;
      position_labels.push_back(index_num);
      num_output_dims++;
    }
  } else {                                   // create a right hand side: the indices that occur exactly once in alphabetic order
    for (size_t idx = 0; idx < number_of_letters; idx++) {
      if (number_of_occurrences[idx] == 1) {
	sorted_position[idx] = num_output_dims;
	position_labels.push_back(idx);
	num_output_dims++;
      }
    }
  }
  // now we assign the sorted_positions ((letter) index -> dimension in Tensors) and position_labels (dimensions in Tensors -> index)
  // for the non-output indices - those that are eventually summed over
  int64_t position = num_output_dims;            // we now determine the porder of the remaining indices (in so far they are in the equation)
  for (size_t idx = 0; idx < number_of_letters; idx++) {
    if ((number_of_occurrences[idx] > 0) && (sorted_position[idx]==-1)) {
      sorted_position[idx] = position;
      position_labels.push_back(idx);
      position++;
    }
  }
  // we now "homogenize the dimensions", i.e. create all dimensions in each tensor and sort the dimensions according to the mapping in
  // sorted_postition / position_labels
  // after this, all operands will have compatible shapes (i.e. all dimensions are aligned are broadcastable)
  std::vector<Tensor> permuted_ops;
  eqn_stream.clear();
  eqn_stream.seekg(0, std::ios_base::beg);
  for (int64_t op = 0; op < (int64_t) tensors.size(); op++) {
    std::array<int64_t, number_of_letters> axes; // the dimension which the letter refers to in the permuted tensor
    axes.fill(-1);
    std::vector<int64_t> permutation; // permutation for this tensor
    std::getline(eqn_stream, term, ',');
    int64_t dim = 0;
    for (auto &c : term) {
      int64_t index_num = c-'a';
      axes[index_num] = dim;
      dim++;
    }
    for (auto &c : position_labels) {
      if (axes[c] > -1) {
	permutation.push_back(axes[c]);
      }
    }
    permuted_ops.push_back(tensors[op].permute(permutation));
    for (int64_t dim = 0; dim < (int64_t) position_labels.size(); dim++) {
      auto c = position_labels[dim];
      if (axes[c] == -1) {
	permuted_ops.back().unsqueeze_(dim);
      }
    }
  }
  // now we reduce the indices from left to right
  // numpy allows to optimize the path using various
  // algorithms (see eigen_path in numpy docs)
  // we start with the leftmost operator and reduce indices that
  // appear only there
  Tensor result = permuted_ops[0];
  for (int64_t idx = 0; idx < number_of_letters; idx++) {
    if ((last_occurrence[idx] == 0)
	&& (sorted_position[idx]>=num_output_dims)) {
      result = result.sum(sorted_position[idx], true);
    }
  }

  // now we process each tensor using sumproduct_pair
  for (int64_t i = 1; i < (int64_t) permuted_ops.size(); i++) {
    std::vector<int64_t> sum_dims;
    for (int64_t idx = 0; idx < number_of_letters; idx++) {
      if ((last_occurrence[idx] == i)
	  && (sorted_position[idx]>=num_output_dims)) {
	sum_dims.push_back(sorted_position[idx]);
      }
    }
    result = at::native::sumproduct_pair(result, permuted_ops[i], sum_dims, true);
  }
  // finally, we squeeze out all non-result dimensions
  for (int64_t dim = position_labels.size()-1; dim >= num_output_dims; dim--)
    result.squeeze_(dim);
  return result;
}

// _trilinear computes a trilinear einstein sum with an unrolled dimension
// the result is `(i1.unsqueeze(expand1)*i2.unsqueeze(expand2)*i2.unsqueeze(expand3)).sum(sumdim)`
// the computation is unrolled in the unroll_dim dimension
// its main purpose is to unify the computations in bilinear and bilinear_backward
Tensor _trilinear(const Tensor& i1_, const Tensor& i2_, const Tensor& i3_,
		  IntList expand1_, IntList expand2_, IntList expand3_,
		  IntList sumdim_, int64_t unroll_dim) {
  int64_t total_dim = i1_.dim()+expand1_.size();
  AT_ASSERT((unroll_dim >= 0) && (unroll_dim < total_dim), "unroll_dim must be in [0,%zd]", total_dim-1);
  auto expand1 = dim_list_to_bitset(expand1_, total_dim);
  auto expand2 = dim_list_to_bitset(expand2_, total_dim);
  auto expand3 = dim_list_to_bitset(expand3_, total_dim);
  auto sumdim  = dim_list_to_bitset(sumdim_,  total_dim);
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

  auto output = i1.type().tensor(output_size).zero_();
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
  AT_ASSERT(input1.dim() == input2.dim(), "bilinear(): input dimensions do not match: got %lld and %lld",
            (long long)input1.dim(), (long long)input2.dim());
  for (int64_t i = 0; i < input1.dim() - 1; i++) {
    AT_ASSERT(input1.size(i) == input2.size(i),
              "bilinear(): input batch dimensions do not match at dim %lld: got %lld and %lld",
              (long long)i, (long long)input1.size(i), (long long)input2.size(i));
  }
  AT_ASSERT(input1.size(input1.dim() - 1) == weight.size(1),
            "bilinear(): input1 size does not match weight size: got %lld but expected %lld",
            (long long)input1.size(input1.dim() - 1), (long long)weight.size(1));
  AT_ASSERT(input2.size(input2.dim() - 1) == weight.size(2),
            "bilinear(): input2 size does not match weight size: got %lld but expected %lld",
            (long long)input2.size(input2.dim() - 1), (long long)weight.size(2));
  AT_ASSERT(!bias.defined() || bias.size(0) == weight.size(0),
            "bilinear(): bias size does not match weight size: got %lld but expected %lld",
            (long long)bias.size(0), (long long)weight.size(0));

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

}}  // namespace at::native
