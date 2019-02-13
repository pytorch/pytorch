#include "caffe2/core/logging.h"
#include "caffe2/utils/cpuid.h"
#include "l2_minimization.h"

#include <cassert>
#include <cmath>
#include <limits>

#include <x86intrin.h>

using namespace std;

namespace dnnlowp {

#undef NDEBUG

static float
GetNorm(float begin, float end, float density, NormMinimization::Kind kind) {
  float norm = 0;

  // assume values are uniformly distributed within each histogram bin
  if (NormMinimization::L2 == kind) {
    // err = density * (integral_{begin, end} x^2)
    //     = density * (end^3 - begin^3) / 3
    norm = (end * end * end - begin * begin * begin) / 3;
    // for begin = d/2 and end = -d/2, this leads to d^3/12
  } else {
    // err = density * (integral_{begin, end} |x|)
    //     = density * (end^2 - begin^2) / 2
    float left_begin = std::min(0.0f, begin);
    float left_end = std::min(0.0f, end);
    assert(left_begin * left_begin >= left_end * left_end);
    norm += (left_begin * left_begin - left_end * left_end) / 2;

    float right_begin = std::max(0.0f, begin);
    float right_end = std::max(0.0f, end);
    assert(right_end * right_end >= right_begin * right_begin);
    norm += (right_end * right_end - right_begin * right_begin) / 2;
  }

  return density * norm;
}

// Filter out outliers in input distributions
// Exploit the input distributions for the quick search
TensorQuantizationParams NormMinimization::NonlinearQuantizationParamsSearch(
    const Histogram& hist,
    bool preserve_sparsity,
    int precision) {
  if (preserve_sparsity) {
    VLOG(2) << "l2_approx with symmetric quantization falls back to L2";
    return ChooseQuantizationParams(hist, preserve_sparsity, precision);
  }
  VLOG(2) << "Using the nonlinear quantile search";
  const vector<uint64_t> bins = *hist.GetHistogram();
  int nbins = bins.size();
  int dst_nbins = 1 << precision;
  float min = hist.Min(), max = hist.Max();
  assert(min <= 0.f);
  assert(max >= 0.f);
  double bin_width = (max - min) / nbins;

  // calculate the CDF
  uint64_t total = 0;
  for (uint64_t x : bins) {
    total += x;
  }
  vector<uint64_t> CDF;
  uint64_t sum = 0;
  for (uint64_t x : bins) {
    sum += x;
    CDF.push_back(sum);
  }

  double stepsize = 0.00001; // experiment on the granularity
  double alpha = 0.0f, beta = 1.0f; // lowerbound and upperbound
  int start_bin = 0;
  int end_bin = nbins - 1;
  double norm_min = numeric_limits<double>::max();

  while (alpha < beta) {
    // find the next step
    double next_alpha = alpha + stepsize;
    double next_beta = beta - stepsize;

    // find the left and right bins between the quantile bounds
    int i = start_bin, j = end_bin;
    while (i < end_bin && CDF[i] < next_alpha * total)
      i++;
    while (j > start_bin && CDF[j] > next_beta * total)
      j--;

    // decide the next move
    // cout << i << ", " << j << endl;
    int next_start_bin = start_bin, next_end_bin = end_bin;
    if ((i - start_bin) > (end_bin - j)) {
      // move the start_bin
      next_start_bin = i;
      alpha = next_alpha;
    } else {
      // move the end_bin
      next_end_bin = j;
      beta = next_beta;
    }

    if (next_start_bin == start_bin && next_end_bin == end_bin)
      continue;
    // calculate the norm
    double norm = 0;
    double dst_bin_width =
        bin_width * (next_end_bin - next_start_bin + 1) / dst_nbins;

    // go over each histogram bin and accumulate errors
    for (int src_bin = 0; src_bin < nbins; ++src_bin) {
      // distances from the beginning of first dst_bin to the beginning and
      // end of src_bin
      double src_bin_begin = (src_bin - next_start_bin) * bin_width;
      double src_bin_end = src_bin_begin + bin_width;

      // which dst_bins the beginning and end of src_bin belong to?
      int dst_bin_of_begin = std::min(
          (1 << precision) - 1.,
          std::max(0., floor(src_bin_begin / dst_bin_width)));
      int dst_bin_of_end = std::min(
          (1 << precision) - 1.,
          std::max(0., floor(src_bin_end / dst_bin_width)));

      double dst_bin_of_begin_center =
          dst_bin_of_begin * dst_bin_width + dst_bin_width / 2;
      double density = bins[src_bin] / bin_width;
      if (dst_bin_of_begin == dst_bin_of_end) {
        // if src_bin is entirely within 1 dst_bin
        double delta_begin = src_bin_begin - dst_bin_of_begin_center;
        double delta_end = src_bin_end - dst_bin_of_begin_center;
        norm += GetNorm(delta_begin, delta_end, density, kind_);
      } else {
        double delta_begin = src_bin_begin - dst_bin_of_begin_center;
        double delta_end = dst_bin_width / 2;
        norm += GetNorm(delta_begin, delta_end, density, kind_);

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) *
            GetNorm(-dst_bin_width / 2, dst_bin_width / 2, density, kind_);

        double dst_bin_of_end_center =
            dst_bin_of_end * dst_bin_width + dst_bin_width / 2;
        delta_begin = -dst_bin_width / 2;
        delta_end = src_bin_end - dst_bin_of_end_center;
        norm += GetNorm(delta_begin, delta_end, density, kind_);
      }
    }
    if (norm > norm_min)
      break;
    norm_min = norm;
    start_bin = next_start_bin;
    end_bin = next_end_bin;
  }
  VLOG(2) << "best quantization range " << start_bin << "," << end_bin + 1
          << "," << norm_min;

  double selected_sum = 0;
  for (int i = start_bin; i < end_bin + 1; ++i) {
    selected_sum += bins[i];
  }
  VLOG(2) << "best quantization range covers "
          << (double)selected_sum / total * 100 << " %%";

  min = hist.Min() + bin_width * start_bin;
  max = hist.Min() + bin_width * (end_bin + 1);

  QuantizationFactory* qfactory = QuantizationFactory::GetDefaultInstance();
  return qfactory->ChooseQuantizationParams(
      min, max, precision, preserve_sparsity);
}

TensorQuantizationParams NormMinimization::ChooseQuantizationParams(
    const Histogram& hist,
    bool preserve_sparsity,
    int precision) {
  VLOG(2) << "Using the brute force search";
  const vector<uint64_t> bins = *hist.GetHistogram();
  int nbins = bins.size();
  vector<float> bins_f(nbins);
  for (int i = 0; i < nbins; ++i) {
    bins_f[i] = bins[i];
  }
  int dst_nbins = 1 << precision;
  float min = hist.Min(), max = hist.Max();
  assert(min <= 0.f);
  assert(max >= 0.f);
  float bin_width = (max - min) / nbins;
  int zero_bin = round(-min / bin_width);

  vector<pair<int, float>> best_start_bins(nbins + 1);

  // Look at mapping [start_bin, start_bin + nbins_selected) to
  // [0, 1 << precision) for every (start_bin, nbins_selected) combination and
  // pick the one with smallest L2 quantization error
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int nbins_selected = 1; nbins_selected <= nbins; ++nbins_selected) {
    float norm_min = numeric_limits<float>::max();
    int best_start_bin = 0;

    int start_bin_begin = 0, start_bin_end = nbins - nbins_selected + 1;
    if (preserve_sparsity) {
      // when preserving sparsity we only check the range
      // starting from 0 (when min is 0) or symmetric around 0.
      if (min == 0) {
        start_bin_begin = 0;
        start_bin_end = 1;
      } else {
        start_bin_begin = zero_bin - nbins_selected / 2;
        start_bin_end = start_bin_begin + 1;
      }
    }

    float dst_bin_width = bin_width * nbins_selected / dst_nbins;

    int start_bin;
    for (start_bin = start_bin_begin; start_bin < start_bin_end; ++start_bin) {
      float norm = 0;

      // go over each histogram bin and accumulate errors
      caffe2::CpuId cpuid = caffe2::GetCpuId();
      if (kind_ == NormMinimization::L2 && cpuid.avx2() && cpuid.fma()) {
        norm = internal::L2MinimizationKernelAVX2(
            precision,
            bins_f.data(),
            nbins,
            bin_width,
            dst_bin_width,
            start_bin);
      } else {
        for (int src_bin = 0; src_bin < nbins; ++src_bin) {
          // distances from the beginning of first dst_bin to the beginning and
          // end of src_bin
          float src_bin_begin = (src_bin - start_bin) * bin_width;
          float src_bin_end = src_bin_begin + bin_width;

          // which dst_bins the beginning and end of src_bin belong to?
          int dst_bin_of_begin = std::min(
              (1 << precision) - 1.0f,
              std::max(0.0f, floorf(src_bin_begin / dst_bin_width)));
          int dst_bin_of_end = std::min(
              (1 << precision) - 1.0f,
              std::max(0.0f, floorf(src_bin_end / dst_bin_width)));

          float dst_bin_of_begin_center =
              dst_bin_of_begin * dst_bin_width + dst_bin_width / 2;
          float density = bins[src_bin] / bin_width;
          float delta_begin = src_bin_begin - dst_bin_of_begin_center;
          if (dst_bin_of_begin == dst_bin_of_end) {
            // if src_bin is entirely within 1 dst_bin
            float delta_end = src_bin_end - dst_bin_of_begin_center;
            norm += GetNorm(delta_begin, delta_end, density, kind_);
          } else {
            float delta_end = dst_bin_width / 2;
            norm += GetNorm(delta_begin, delta_end, density, kind_);

            norm += (dst_bin_of_end - dst_bin_of_begin - 1) *
                GetNorm(-dst_bin_width / 2, dst_bin_width / 2, density, kind_);

            float dst_bin_of_end_center =
                dst_bin_of_end * dst_bin_width + dst_bin_width / 2;
            delta_begin = -dst_bin_width / 2;
            delta_end = src_bin_end - dst_bin_of_end_center;
            norm += GetNorm(delta_begin, delta_end, density, kind_);
          }
        }
      }

      if (norm < norm_min) {
        norm_min = norm;
        best_start_bin = start_bin;
      }
    } // for each start_bin

    best_start_bins[nbins_selected] = {best_start_bin, norm_min};
  } // for each nbins_selected

  float norm_min = numeric_limits<float>::max();
  int best_nbins_selected = 1, best_start_bin = 0;
  for (int nbins_selected = 1; nbins_selected <= nbins; ++nbins_selected) {
    float norm = best_start_bins[nbins_selected].second;
    if (norm < norm_min) {
      norm_min = norm;
      best_start_bin = best_start_bins[nbins_selected].first;
      best_nbins_selected = nbins_selected;
    }
  }

  float total_sum = 0;
  for (int i = 0; i < bins.size(); ++i) {
    total_sum += bins[i];
  }
  float selected_sum = 0;
  int i_begin = std::max(0, best_start_bin);
  int i_end = std::min(nbins, best_start_bin + best_nbins_selected);
  for (int i = i_begin; i < i_end; ++i) {
    selected_sum += bins[i];
  }
  VLOG(2) << "best quantization range covers " << selected_sum / total_sum * 100
          << " %%";

  min = hist.Min() + bin_width * (best_start_bin);
  max = hist.Min() + bin_width * (best_start_bin + best_nbins_selected);

  QuantizationFactory* qfactory = QuantizationFactory::GetDefaultInstance();
  return qfactory->ChooseQuantizationParams(
      min, max, precision, preserve_sparsity);
} // ChooseQuantizationParams

} // namespace dnnlowp
