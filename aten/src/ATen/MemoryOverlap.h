#pragma once

#include <ATen/ATen.h>

namespace at {

// NOTE [ Detecting Memory Overlap Within A Strided Tensor ]
//
// Checking memory overlap within a strided tensor is the special case of
// detecting memory overlap of two strided tensors, where the two tensors start
// at the same memory address. The later is HARD (see #8212).
//
// But even this special case isn't simple. This note describes a check for a
// even more constrained simple case where we can be certain that there is no
// overlap.
//
// The checking algorithm can be described as:
//   0. Return [ pass check ] if any dimension has size 0
//   1. Ignore all dimensions that have size 1
//   2. If no remaining dimensions, return [ pass check ]
//   3. Sort the remaining dimensions according to the strides decreasingly
//   4. Check that for each dimension k,
//
//           stride[k] > \sum_{ i > k } (size[i] - 1) * stride[i]
//
//      That is equivalent to, after reordering the dimensions so strides are
//      in decreasing order, checking that stride of each dimension is larger
//      than the maximum memory offset in a slice at that dimension.
//
// Obviously this check passes for contiguous tensors ( the dimensions will be
// already sorted with LHS = stride[0] = \prod size[i] being exactly 1 larger
// than RHS ). Similarly, the check passes for tensors contiguous in all but
// the last dimension, and LHS = stride[0] = stride[-1] * \prod size[i] being
// exactly stride[-1] larger than RHS. (*)
//
// We will show that these view operations, including all our view operations
// *except for* general as_strided and unfold, also preserve this invariant:
//
//  alias:      Obviously preserves
//
//  expand:     All changed dimensions are removed in step (1)
//
//  view:       Consider the input dimensions as grouped into consecutive
//              dimension "blocks", where dimensions are contiguous in each one.
//              one. view only works when the output dimensions can also be
//              grouped into the same consecutive blocks of same ordering.
//
//              NB: this means that the number of elements and stride of the
//                  last dimension in each block is the same in input and
//                  output. (**)
//
//              Notation:
//                Consider a single such block B,
//                    ... B_prev[-1]], [ B[0], ..., B[i], ..., B[k] = B[-1] ], [ B_next[0], ...
//                                start--^^^^                  ^^^^^^^^^^^^--end
//                Each B[i] denotes a dimension index such that B[i] = B[0] + i.
//
//              We first show that in a tensor (i.e., input) satisfies the
//              invariant, after sorting, the dimensions within each block
//              still remain consecutive. (***)
//
//                After removing dimensions of size 1, the dimensions within a
//                block is already sorted by strides in descending order. So
//                sorting all dimensions will not change the relative ordering
//                among them.
//
//                Assume that some block B is not consecutive after sorting,
//                i.e., there exists a dimension d between B[0] and B[-1] in
//                sorted order.
//
//                By (*), we know that
//                       stride[B[0]]
//                    =  \sum_{i > 0}   (size[B[i]] - 1) * stride[B[i]] + stride[B[-1]]
//                    <  \sum_{i > 0}   (size[B[i]] - 1) * stride[B[i]] + stride[d]
//                    <= \sum_{i > 0}   (size[B[i]] - 1) * stride[B[i]] + (size[d] - 1) * stride[d]
//                    <= \sum{j > B[0]} (size[j]    - 1) * stride[j],
//
//                where the first <   comes from sorting and
//                      the second <= comes from the fact that dimension d
//                                               exists after step (1) and
//                                               thus must have size greater
//                                               than 1
//                      the third  <= comes from the fact that each term in
//                                               the sum is non-negative
//
//                Then we have a countradiction as the invariant must not be
//                satisfied at B[0]. So the original proposition is true.
//
//              Now that we established the above claim (***), we consider the
//              view operation as first sorting the dimensions (i.e., blocks),
//              apply the original view (since it only cares dimensions being
//              consecutive and contiguous withtin each block), and then undo
//              the sort.
//
//              Consider a single block B in the output,
//                  ... ], [ B[0], ..., B[i], ..., B[k] = B[-1] ], [ ...
//                    start--^^^^                  ^^^^^^^^^^^^--end
//
//              By (*), we know that for all i
//                  stride[i] = stride[B[-1]] +
//                                \sum_{j=i+1}^{k} (size[B[j]] - 1) * stride[B[j]]
//
//              Then the invariant is obviously satisfied at every dimension
//              in this block if it is satisfied at dimnesion B[-1]. It only
//              remains to show that it is satisfied at the last dimension in
//              each block.
//
//              Since the same blocks are present in both input and output
//              with the same ordering, we will abuse the notation in the
//              following statements.
//
//              By (*), we know that the following holds for both input and
//              output, for any block B:
//                    \sum_{i > B[-1]} (size[i] - 1) * stride[i]
//                  = \sum_{block B' after B} \prod_{j in B'} size[B[j]] * stride[B'[-1]]
//                  = \sum_{block B' after B} numel(B') * stride[B'[-1]].
//                    ^^^^^^^^^^^^^^^^^^^^^^^|^^^^^^^^^^^^^^^^^^^^^^^^^^
//              By (**), we know that, this quantity in the above equation
//              remains the same in input and output. So both
//                  \sum_{i > B[-1]} (size[i] - 1) * stride[i]
//              and
//                  stride[B[-1]]
//              are the same in input and output.
//
//              These two quantities are exactly the LHS and RHS of the
//              invariant inequality. Since by assumption the invariant is
//              satisfied in input at B[-1], it is also satisfied in output at
//              B[-1]. This concludes the proof.
//
//  squeeze:    Special case of view
//
//  unsqueeze:  Special case of view
//
//  slice:      Consider slicing dimension i with step = k >= 1.
//
//              Let stride' and size' be the output strides and sizes. We have
//
//                  stride'[i] = k * stride[i]
//                  size'[i] <= floor(size[i] / k)
//
//              If size'[i] = 1, invariant is obviously satisfied as we are
//              just removing a dimension (afte step (1)).
//
//              Assume size'[i] > 1.
//
//              By assumption, the invariant is satisfied at every dimension
//              in input.
//
//              For any dimension j, if stride[j] > stride[i], we have
//                  stride'[j] =  stride[j]
//                             >  (size[i] - 1) * stride[i]
//                             =  (size[i] / k * k - 1) * k * stride[i] / k
//                             =  (size[i] / k - 1 / k) * stride'[i]
//                             >= (size'[i]    - 1 / k) * stride'[i]
//                             >= stride'[i].
//
//              If stride[j] < stride[i], we have
//                  stride'[j] = stride[j] < stride[i] <= stride'[i].
//
//              So the sorting order remains unchanged after slice.
//
//              Since
//                     (size'[i] - 1) * stride'[i]
//                  =  (floor(size[i] / k) - 1) * k * stride[i]
//                  <= (size[i] / k - 1) * k * stride[i]
//                  =  (size[i] - k) * stride[i]
//                  <= (size[i] - 1) * * stride[i],
//              the term from this dimension i in the invariant inequality at
//              other dimensions can only decrease after slice. So the
//              invariant is preserved.
//
//  narrow:     Special case of slice
//
//  select:     narrow + squeeze
//
//  permute:    Sorting makes permutation of dimensions irrelevant
//
//  transpose:  Sorting makes swapping dimensions irrelevant
//
//  diagonal:   Effectively merging two dimensions i and j into a new
//              dimension k s.t.
//                  stride'[k] =  stride[i] + stride[j]
//                  size'[k]   <= min(size[i], size[j]),
//              where stride and size are on the input, and stride' and size'
//              are on the output.
//
//              Assuming that size[i] > 1 and size[j] > 1. If any has size 1,
//              then this is unsqueeze on that dimension.
//
//              WLOG, say stride[i] >= stride[j].
//
//              Each dimension d in input with stride[d] > stride[j] has
//                  stride'[d] =  stride[d]
//                             >  (size[i] - 1) * stride[i] + (size[j] - 1) * stride[j]
//                             >= stride[i] + stride[j]
//                             =  stride[k].
//              So, considering the sorted dimensions, this is effectively
//              removing i, and replacing j with k.
//
//              For dimensions d with stride[i] < stride[d] < stride[j], the
//              term from dimension i is removed in the invariant inequality.
//              For dimensions d with stride[d] > stride[j], we have
//                     (size'[k] - 1) * stride'[k]
//                  <= (min(size[i], size[j]) - 1) * (stride[i] + stride[j])
//                  <= (size[i] - 1) * stride[i] + (size[j] - 1) * stride[j],
//              so the term from i and j in the invariant can only decrease.
//
//              So this is generally relaxing the constraint, and thus it
//              preserves it.

// This implements steps (2)~(4) of the algorithm in
// NOTE [ Detecting Memory Overlap Within A Strided Tensor ]
// We are breaking the implementation for specific requirements as what is
// needed as a helper for as_strided_backward
inline bool maybe_overlapping_memory(IntArrayRef sizes, IntArrayRef strides) {
  if (sizes.size() > 0) {
    std::vector<std::size_t> argsort(sizes.size());
    std::iota(argsort.begin(), argsort.end(), 0);
    std::sort(argsort.begin(), argsort.end(),
        [&](std::size_t i, std::size_t j){ return strides[i] < strides[j]; });

    int64_t max_index_in_slice = 0;
    for (auto i : argsort) {
      auto stride_ = strides[i];
      if (stride_ <= max_index_in_slice) {
        return true;
      }
      max_index_in_slice += stride_ * (sizes[i] - 1);
    }
  }
  return false;
}

// MemOverlap: Whether or not there is memory overlap
//
// NO: Absolutely no memory overlap
// YES: Absolutely yes memory overlap
// TOO_HARD: There might be memory overlap, but it was too expensive to compute.
//
// NB: Please update the python test for these if you renumber them.
enum class MemOverlap { NO, YES, TOO_HARD };

enum class MemOverlapStatus { FULL, PARTIAL, NO, TOO_HARD };

CAFFE2_API MemOverlap has_internal_overlap(const Tensor& t);
CAFFE2_API MemOverlap has_internal_overlap(TensorImpl* t);

CAFFE2_API void assert_no_internal_overlap(const Tensor& t);
CAFFE2_API void assert_no_internal_overlap(TensorImpl* t);

MemOverlapStatus get_overlap_status(const Tensor& a, const Tensor& b);
MemOverlapStatus get_overlap_status(TensorImpl* a, TensorImpl* b);

void assert_no_partial_overlap(const Tensor& a, const Tensor& b);
void assert_no_partial_overlap(TensorImpl* a, TensorImpl* b);

}
