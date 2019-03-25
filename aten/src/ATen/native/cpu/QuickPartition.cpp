// AVX2 optimized partition algorithm
// Based on the following paper:
// Shay Gueron, Vlad Krasnov, Fast Quicksort Implementation Using AVX Instructions, The Computer Journal, Volume 59, Issue 1, January 2016, Pages 83–90, https://doi.org/10.1093/comjnl/bxv063

#include <aten/src/ATen/native/cpu/QuickPartition.h>
#include <stdint.h>
#if CPU_CAPABILITY_AVX2
#include <aten/src/ATen/cpu/vec256/vec256.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <algorithm>
#include <limits>
#include <cmath>

using at::vec256::Vec256;
using at::vec256::permute;
using at::vec256::int_same_size_t;


template <typename T>
struct Table;

template <>
struct Table<float> {
  static const uint32_t __attribute__((aligned(256))) t[256*8];
};

/* These values represent the reordering indices for a given comparison result
 * that will put elements <= than the pivot at the beginning of
 * the vector, and elements >= the pivot at the end (or vice versa for a descending sort)
 *
 * For example, if the pivot is 5.2, given a vector (1.0, 6.3, 7.9, 2.8, 4.2, 6.7, 0.3, 2,3),
 * the pointwise comparison result is (0, 1, 1, 0, 0, 1, 0, 0).  Interpreted as a bitwise-little-endian binary value,
 * this represents the integer 38.
 *
 * Then Table[38*8] through Table[38*8 + 7] inclusive are the indices (0,3,4,6,7,1,2,5),
 * which represent the permutation that takes the input vector to
 * (1.0, 2.8, 4.2, 0.3, 2.3, 6.3, 7.9, 6.7).
 */
const uint32_t __attribute__((aligned(256))) Table<float>::t[] = {
      0,1,2,3,4,5,6,7,
      1,2,3,4,5,6,7,0,
      0,2,3,4,5,6,7,1,
      2,3,4,5,6,7,0,1,
      0,1,3,4,5,6,7,2,
      1,3,4,5,6,7,0,2,
      0,3,4,5,6,7,1,2,
      3,4,5,6,7,0,1,2,
      0,1,2,4,5,6,7,3,
      1,2,4,5,6,7,0,3,
      0,2,4,5,6,7,1,3,
      2,4,5,6,7,0,1,3,
      0,1,4,5,6,7,2,3,
      1,4,5,6,7,0,2,3,
      0,4,5,6,7,1,2,3,
      4,5,6,7,0,1,2,3,
      0,1,2,3,5,6,7,4,
      1,2,3,5,6,7,0,4,
      0,2,3,5,6,7,1,4,
      2,3,5,6,7,0,1,4,
      0,1,3,5,6,7,2,4,
      1,3,5,6,7,0,2,4,
      0,3,5,6,7,1,2,4,
      3,5,6,7,0,1,2,4,
      0,1,2,5,6,7,3,4,
      1,2,5,6,7,0,3,4,
      0,2,5,6,7,1,3,4,
      2,5,6,7,0,1,3,4,
      0,1,5,6,7,2,3,4,
      1,5,6,7,0,2,3,4,
      0,5,6,7,1,2,3,4,
      5,6,7,0,1,2,3,4,
      0,1,2,3,4,6,7,5,
      1,2,3,4,6,7,0,5,
      0,2,3,4,6,7,1,5,
      2,3,4,6,7,0,1,5,
      0,1,3,4,6,7,2,5,
      1,3,4,6,7,0,2,5,
      0,3,4,6,7,1,2,5,
      3,4,6,7,0,1,2,5,
      0,1,2,4,6,7,3,5,
      1,2,4,6,7,0,3,5,
      0,2,4,6,7,1,3,5,
      2,4,6,7,0,1,3,5,
      0,1,4,6,7,2,3,5,
      1,4,6,7,0,2,3,5,
      0,4,6,7,1,2,3,5,
      4,6,7,0,1,2,3,5,
      0,1,2,3,6,7,4,5,
      1,2,3,6,7,0,4,5,
      0,2,3,6,7,1,4,5,
      2,3,6,7,0,1,4,5,
      0,1,3,6,7,2,4,5,
      1,3,6,7,0,2,4,5,
      0,3,6,7,1,2,4,5,
      3,6,7,0,1,2,4,5,
      0,1,2,6,7,3,4,5,
      1,2,6,7,0,3,4,5,
      0,2,6,7,1,3,4,5,
      2,6,7,0,1,3,4,5,
      0,1,6,7,2,3,4,5,
      1,6,7,0,2,3,4,5,
      0,6,7,1,2,3,4,5,
      6,7,0,1,2,3,4,5,
      0,1,2,3,4,5,7,6,
      1,2,3,4,5,7,0,6,
      0,2,3,4,5,7,1,6,
      2,3,4,5,7,0,1,6,
      0,1,3,4,5,7,2,6,
      1,3,4,5,7,0,2,6,
      0,3,4,5,7,1,2,6,
      3,4,5,7,0,1,2,6,
      0,1,2,4,5,7,3,6,
      1,2,4,5,7,0,3,6,
      0,2,4,5,7,1,3,6,
      2,4,5,7,0,1,3,6,
      0,1,4,5,7,2,3,6,
      1,4,5,7,0,2,3,6,
      0,4,5,7,1,2,3,6,
      4,5,7,0,1,2,3,6,
      0,1,2,3,5,7,4,6,
      1,2,3,5,7,0,4,6,
      0,2,3,5,7,1,4,6,
      2,3,5,7,0,1,4,6,
      0,1,3,5,7,2,4,6,
      1,3,5,7,0,2,4,6,
      0,3,5,7,1,2,4,6,
      3,5,7,0,1,2,4,6,
      0,1,2,5,7,3,4,6,
      1,2,5,7,0,3,4,6,
      0,2,5,7,1,3,4,6,
      2,5,7,0,1,3,4,6,
      0,1,5,7,2,3,4,6,
      1,5,7,0,2,3,4,6,
      0,5,7,1,2,3,4,6,
      5,7,0,1,2,3,4,6,
      0,1,2,3,4,7,5,6,
      1,2,3,4,7,0,5,6,
      0,2,3,4,7,1,5,6,
      2,3,4,7,0,1,5,6,
      0,1,3,4,7,2,5,6,
      1,3,4,7,0,2,5,6,
      0,3,4,7,1,2,5,6,
      3,4,7,0,1,2,5,6,
      0,1,2,4,7,3,5,6,
      1,2,4,7,0,3,5,6,
      0,2,4,7,1,3,5,6,
      2,4,7,0,1,3,5,6,
      0,1,4,7,2,3,5,6,
      1,4,7,0,2,3,5,6,
      0,4,7,1,2,3,5,6,
      4,7,0,1,2,3,5,6,
      0,1,2,3,7,4,5,6,
      1,2,3,7,0,4,5,6,
      0,2,3,7,1,4,5,6,
      2,3,7,0,1,4,5,6,
      0,1,3,7,2,4,5,6,
      1,3,7,0,2,4,5,6,
      0,3,7,1,2,4,5,6,
      3,7,0,1,2,4,5,6,
      0,1,2,7,3,4,5,6,
      1,2,7,0,3,4,5,6,
      0,2,7,1,3,4,5,6,
      2,7,0,1,3,4,5,6,
      0,1,7,2,3,4,5,6,
      1,7,0,2,3,4,5,6,
      0,7,1,2,3,4,5,6,
      7,0,1,2,3,4,5,6,
      0,1,2,3,4,5,6,7,
      1,2,3,4,5,6,0,7,
      0,2,3,4,5,6,1,7,
      2,3,4,5,6,0,1,7,
      0,1,3,4,5,6,2,7,
      1,3,4,5,6,0,2,7,
      0,3,4,5,6,1,2,7,
      3,4,5,6,0,1,2,7,
      0,1,2,4,5,6,3,7,
      1,2,4,5,6,0,3,7,
      0,2,4,5,6,1,3,7,
      2,4,5,6,0,1,3,7,
      0,1,4,5,6,2,3,7,
      1,4,5,6,0,2,3,7,
      0,4,5,6,1,2,3,7,
      4,5,6,0,1,2,3,7,
      0,1,2,3,5,6,4,7,
      1,2,3,5,6,0,4,7,
      0,2,3,5,6,1,4,7,
      2,3,5,6,0,1,4,7,
      0,1,3,5,6,2,4,7,
      1,3,5,6,0,2,4,7,
      0,3,5,6,1,2,4,7,
      3,5,6,0,1,2,4,7,
      0,1,2,5,6,3,4,7,
      1,2,5,6,0,3,4,7,
      0,2,5,6,1,3,4,7,
      2,5,6,0,1,3,4,7,
      0,1,5,6,2,3,4,7,
      1,5,6,0,2,3,4,7,
      0,5,6,1,2,3,4,7,
      5,6,0,1,2,3,4,7,
      0,1,2,3,4,6,5,7,
      1,2,3,4,6,0,5,7,
      0,2,3,4,6,1,5,7,
      2,3,4,6,0,1,5,7,
      0,1,3,4,6,2,5,7,
      1,3,4,6,0,2,5,7,
      0,3,4,6,1,2,5,7,
      3,4,6,0,1,2,5,7,
      0,1,2,4,6,3,5,7,
      1,2,4,6,0,3,5,7,
      0,2,4,6,1,3,5,7,
      2,4,6,0,1,3,5,7,
      0,1,4,6,2,3,5,7,
      1,4,6,0,2,3,5,7,
      0,4,6,1,2,3,5,7,
      4,6,0,1,2,3,5,7,
      0,1,2,3,6,4,5,7,
      1,2,3,6,0,4,5,7,
      0,2,3,6,1,4,5,7,
      2,3,6,0,1,4,5,7,
      0,1,3,6,2,4,5,7,
      1,3,6,0,2,4,5,7,
      0,3,6,1,2,4,5,7,
      3,6,0,1,2,4,5,7,
      0,1,2,6,3,4,5,7,
      1,2,6,0,3,4,5,7,
      0,2,6,1,3,4,5,7,
      2,6,0,1,3,4,5,7,
      0,1,6,2,3,4,5,7,
      1,6,0,2,3,4,5,7,
      0,6,1,2,3,4,5,7,
      6,0,1,2,3,4,5,7,
      0,1,2,3,4,5,6,7,
      1,2,3,4,5,0,6,7,
      0,2,3,4,5,1,6,7,
      2,3,4,5,0,1,6,7,
      0,1,3,4,5,2,6,7,
      1,3,4,5,0,2,6,7,
      0,3,4,5,1,2,6,7,
      3,4,5,0,1,2,6,7,
      0,1,2,4,5,3,6,7,
      1,2,4,5,0,3,6,7,
      0,2,4,5,1,3,6,7,
      2,4,5,0,1,3,6,7,
      0,1,4,5,2,3,6,7,
      1,4,5,0,2,3,6,7,
      0,4,5,1,2,3,6,7,
      4,5,0,1,2,3,6,7,
      0,1,2,3,5,4,6,7,
      1,2,3,5,0,4,6,7,
      0,2,3,5,1,4,6,7,
      2,3,5,0,1,4,6,7,
      0,1,3,5,2,4,6,7,
      1,3,5,0,2,4,6,7,
      0,3,5,1,2,4,6,7,
      3,5,0,1,2,4,6,7,
      0,1,2,5,3,4,6,7,
      1,2,5,0,3,4,6,7,
      0,2,5,1,3,4,6,7,
      2,5,0,1,3,4,6,7,
      0,1,5,2,3,4,6,7,
      1,5,0,2,3,4,6,7,
      0,5,1,2,3,4,6,7,
      5,0,1,2,3,4,6,7,
      0,1,2,3,4,5,6,7,
      1,2,3,4,0,5,6,7,
      0,2,3,4,1,5,6,7,
      2,3,4,0,1,5,6,7,
      0,1,3,4,2,5,6,7,
      1,3,4,0,2,5,6,7,
      0,3,4,1,2,5,6,7,
      3,4,0,1,2,5,6,7,
      0,1,2,4,3,5,6,7,
      1,2,4,0,3,5,6,7,
      0,2,4,1,3,5,6,7,
      2,4,0,1,3,5,6,7,
      0,1,4,2,3,5,6,7,
      1,4,0,2,3,5,6,7,
      0,4,1,2,3,5,6,7,
      4,0,1,2,3,5,6,7,
      0,1,2,3,4,5,6,7,
      1,2,3,0,4,5,6,7,
      0,2,3,1,4,5,6,7,
      2,3,0,1,4,5,6,7,
      0,1,3,2,4,5,6,7,
      1,3,0,2,4,5,6,7,
      0,3,1,2,4,5,6,7,
      3,0,1,2,4,5,6,7,
      0,1,2,3,4,5,6,7,
      1,2,0,3,4,5,6,7,
      0,2,1,3,4,5,6,7,
      2,0,1,3,4,5,6,7,
      0,1,2,3,4,5,6,7,
      1,0,2,3,4,5,6,7,
      0,1,2,3,4,5,6,7,
      0,1,2,3,4,5,6,7,
};

template <typename T, bool largest>
int_same_size_t<T> nan_partition(T *begin, T *end,
    int_same_size_t<T> *indices) {
  // We consider NaNs to be larger than any element.
  if (!largest) {
    // everything is <= nan, which is the pivot, so the
    // array is already partitioned as-is.
    return 0;
  }
  // otherwise, put NaNs at the beginning and return.
  T *left_out = begin;
  T *right_out = end;

  T *left_in = begin;
  T *right_in = end;

  auto sz = end - begin;

  int_same_size_t<T> *indices_left_in = indices;
  int_same_size_t<T> *indices_right_in = indices + sz;

  int_same_size_t<T> *indices_left_out = indices;
  int_same_size_t<T> *indices_right_out = indices + sz;
  while (true) {
    while (left_out < end && std::isnan(*left_out)) {
      ++left_out;
      ++indices_left_out;
    }
    if (left_out > left_in) {
      left_in = left_out;
      indices_left_in = indices_left_out;
    }
    while (left_in < end && !std::isnan(*left_in)) {
      ++left_in;
      ++indices_left_in;
    }
    if (left_in == end) {
      return left_out - begin;
    }
    std::swap(*left_out, *left_in);
    std::swap(*indices_left_out, *indices_left_in);
  }
}

template <typename T, bool largest>
int_same_size_t<T> qs_partition_inplace_scalar(T *begin, T *end,
    int_same_size_t<T> *indices,
    T pivot) {
  // adapted from libc++ std::partition
  T *left = begin;
  T *right = end;
  int_same_size_t<T> *indices_left = indices;
  int_same_size_t<T> *indices_right = indices + (end - begin);
  while (true) {
    while (true) {
      if (left == right)
        return left - begin;
      if (largest ? (*left <= pivot) : !(*left < pivot))
          break;
      ++left;
      ++indices_left;
    }
    do
    {
      --right;
      --indices_right;
      if (left == right)
        return left - begin;
    } while (largest ? (*right <= pivot) : !(*right < pivot));
    std::swap(*left, *right);
    std::swap(*indices_left, *indices_right);
    ++left;
    ++indices_left;
  }
  return left - begin;
}

template <typename T, bool largest>
int_same_size_t<T> qs_partition_inplace(T *begin, T *end,
    int_same_size_t<T> *indices, // XXX int_same_size_t<T>
    T pivot)
{
  if (std::isnan(pivot)) {
    return nan_partition<float, largest>(begin, end, indices);
  }
  // from now on, we can assume pivot is a number.
  auto nelts = Vec256<T>::size();
  if (end-begin < 3 * nelts) {
    return qs_partition_inplace_scalar<T, largest>(begin, end, indices, pivot);
  }
  using IndT = int_same_size_t<T>; // XXX int_same_size_t<T>;
  Vec256<T> Piv(pivot);

  T *left_out = begin;
  T *right_out = end;

  T *left_in = begin;
  T *right_in = end;

  auto sz = end - begin;

  IndT *indices_left_in = indices;
  IndT *indices_right_in = indices + sz;

  IndT *indices_left_out = indices;
  IndT *indices_right_out = indices + sz;
  auto Data1 = Vec256<T>::loadu(left_in);
  auto Ind1 = Vec256<IndT>::loadu(indices_left_in);
  left_in += nelts;
  indices_left_in += nelts;

  auto Data2 = Vec256<T>::loadu(left_in);
  auto Ind2 = Vec256<IndT>::loadu(indices_left_in);
  left_in += nelts;
  indices_left_in += nelts;

  Vec256<T> Data3;
  Vec256<IndT> Ind3;

  // The main loop, which reads and writes nelts elements at a time
  do {
    // Prepare to write out the data in Data1.
    // Permute it so that (when largest == true) all the elements > the pivot are on the left,
    // and all the elements <= the pivot are on the right. The indices for the permutation
    // come from a lookup table. (there are only 2^8==256 possibilities when nelts=8, so the lookup
    // table is a reasonable size)
    Vec256<T> Comp = largest ? Data1 < Piv : Data1.not_leq(Piv);
    int32_t const mask = Comp.msb_mask();
    auto Perm = Vec256<IndT>::loadu(&(Table<float>::t[nelts*mask]));
    auto Ordered
      = permute(Data1, Perm);
    auto IndicesOrdered
      = permute(Ind1, Perm);
    

    uint8_t to_write_right = __builtin_popcount(mask);
    uint8_t to_write_left = nelts - to_write_right;

    // We are about to write the data on both sides of the buffer.
    // To avoid clobbering future input, we need to make sure that before we do so,
    // there are at least `nelts` free spaces on each side
    // (where "free spaces" means locations that we've already read,
    // but not yet written to, so we can use them for output).
    //
    // At this point, there are always exactly `2 * nelts` free spaces in total,
    // so at least one side will have at least `nelts`.
    // So, to free up space, consume the next chunk of input from whichever side doesn't.
    auto left_space = left_in - left_out;
    if (left_space < nelts) {
      Data3 = Vec256<T>::loadu(left_in);
      Ind3 = Vec256<IndT>::loadu(indices_left_in);
      left_in += nelts;
      indices_left_in += nelts;
    } else {
      right_in -= nelts;
      indices_right_in -= nelts;
      Data3 = Vec256<T>::loadu(right_in);
      Ind3 = Vec256<IndT>::loadu(indices_right_in);
    }

    // now write the output (plus some garbage) on both sides.
    Ordered.store(right_out - nelts);
    Ordered.store(left_out);
    IndicesOrdered.store(indices_right_out - nelts);
    IndicesOrdered.store(indices_left_out);

    // update the output pointers depending on how much of the output above was real as opposed to garbage.
    left_out += to_write_left;
    right_out -= to_write_right;
    indices_left_out += to_write_left;
    indices_right_out -= to_write_right;

    Data1 = Data2;
    Data2 = Data3;
    Ind1 = Ind2;
    Ind2 = Ind3;
  } while (right_in - left_in >= nelts);
  // Main loop done.

  // Now we have `2 * nelts` pieces of data still to use, in Data1 and Data2,
  // and we also have some amount in [0, nelts) of data still to read,
  // which we process in a normal scalar way.
  while (left_in != right_in) {
    T val;
    IndT ind;
    if (left_in <= left_out) {
      val = *(--right_in);
      ind = *(--indices_right_in);
    } else {
      val = *left_in++;
      ind = *indices_left_in++;
    }
    bool valShouldGoLeft = largest ? !(pivot <= val) : (val < pivot);
    if (valShouldGoLeft) {
      *left_out++ = val;
      *indices_left_out++ = ind;
    } else {
      *(--right_out) = val;
      *(--indices_right_out) = ind;
    }
  }

  // Now all data has been read, and all but 2 * nelts
  // has been written. It is in Data1 and Data2. Write it out
  // in the same way as in the main loop.
  for (int i = 0; i < 2; ++i) {
    Vec256<T> Comp = largest ? Data1 < Piv : Data1 > Piv;
    int32_t const mask = Comp.msb_mask();
    auto Perm = Vec256<IndT>::loadu(&(Table<float>::t[nelts*mask]));
    auto Ordered
      = permute(Data1, Perm);
    auto IndicesOrdered
      = permute(Ind1, Perm);

    uint8_t to_write_right = __builtin_popcount(mask);
    uint8_t to_write_left = nelts - to_write_right;

    Ordered.store(right_out - nelts);
    Ordered.store(left_out);
    IndicesOrdered.store(indices_right_out - nelts);
    IndicesOrdered.store(indices_left_out);

    left_out += to_write_left;
    right_out -= to_write_right;
    indices_left_out += to_write_left;
    indices_right_out -= to_write_right;

    Data1 = Data2;
    Ind1 = Ind2;
  }
  // We are done.
  return left_out - begin;
}


int32_t at::native::vec_qs_partition_inplace(float *begin, float *end,
    int32_t *indices,
    float pivot,
    bool largest) {
  if (largest) {
    return qs_partition_inplace<float, true>(begin, end, indices, pivot);
  } else {
    return qs_partition_inplace<float, false>(begin, end, indices, pivot);
  }
}
#else

namespace at {
namespace native {
namespace {
// fallback to scalar when we don't have avx2
int32_t vec_qs_partition_inplace(float *begin, float *end,
    int32_t *indices,
    float pivot,
    bool largest) {
  return scalar_partition<float, int32_t>(begin,
      [largest](float x, float y) {
        return largest ? gt_or_nan<float>(x, y)
          : gt_or_nan<float>(y, x);
      },
      [&](int32_t i, int32_t j) {
        std::swap(begin[i], begin[j]);
        std::swap(begin[j], begin[i]);
      },
      0,
      end - begin - 1,
      pivot);
}

} // anonymous namespace
} // namespace at::native
}

#endif // CPU_CAPABILITY_AVX2
