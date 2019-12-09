#include "kl_minimization.h"
#include "caffe2/core/logging.h"

using namespace std;

namespace dnnlowp {

TensorQuantizationParams KLDivergenceMinimization::ChooseQuantizationParams(
    const Histogram& hist,
    bool preserve_sparsity,
    int precision) {
  const vector<uint64_t> bins = *hist.GetHistogram();
  int nbins = bins.size();
  int dst_nbins = 1 << precision;
  float min = hist.Min(), max = hist.Max();
  assert(min <= 0.f);
  assert(max >= 0.f);
  double bin_width = (max - min) / nbins;
  int zero_bin = round(-min / bin_width);

  double total_sum = 0;
  for (int i = 0; i < nbins; ++i) {
    total_sum += bins[i];
  }

  vector<pair<int, double>> best_start_bins(nbins + 1);

  // Look at mapping [start_bin, start_bin + nbins_selected) to
  // [0, 1 << precision) for every (start_bin, nbins_selected) combination and
  // pick the one with smallest KL divergence
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int nbins_selected = 1; nbins_selected <= nbins; ++nbins_selected) {
    // if (nbins_selected % dst_nbins != 0) continue;
    double kl_min = numeric_limits<double>::max();
    int best_start_bin = 0;

    int start_bin_begin = 0, start_bin_end = nbins - nbins_selected + 1;
    if (preserve_sparsity) {
      if (min == 0) {
        start_bin_begin = 0;
        start_bin_end = 1;
      } else {
        start_bin_begin = zero_bin - nbins_selected / 2;
        start_bin_end = start_bin_begin + 1;
      }
    }

    int start_bin;
    for (start_bin = start_bin_begin; start_bin < start_bin_end; ++start_bin) {
      double kl = 0;

      // sum outliers
      uint64_t left_outliers = 0;
      int src_bin;
      for (src_bin = 0; src_bin < start_bin; ++src_bin) {
        left_outliers += bins[src_bin];
      }

      uint64_t right_outliers = 0;
      for (src_bin = start_bin + nbins_selected; src_bin < nbins; ++src_bin) {
        right_outliers += bins[src_bin];
      }

      // each destination bin corresponds to a quantized value
      for (int dst_bin = 0; dst_bin < dst_nbins; ++dst_bin) {
        double non_zero_length = 0;
        double sum = 0;
        double src_bin_begin_not_rounded =
            start_bin + (double)dst_bin * nbins_selected / dst_nbins;
        int src_bin_begin = src_bin_begin_not_rounded;
        double src_bin_end_not_rounded =
            start_bin + (double)(dst_bin + 1) * nbins_selected / dst_nbins;
        int src_bin_end = ceil(src_bin_end_not_rounded);
        for (src_bin = src_bin_begin; src_bin < src_bin_end; ++src_bin) {
          if (src_bin >= 0 && src_bin < nbins) {
            double bin = bins[src_bin];
            double fraction = 1;
            if (src_bin == src_bin_begin && src_bin == src_bin_end - 1) {
              fraction = src_bin_end_not_rounded - src_bin_begin_not_rounded;
            } else if (src_bin == src_bin_begin) {
              fraction = (src_bin_begin + 1) - src_bin_begin_not_rounded;
              assert(fraction >= 0);
            } else if (src_bin == src_bin_end - 1) {
              fraction = src_bin_end_not_rounded - (src_bin_end - 1);
              assert(fraction >= 0);
            }
            bin *= fraction;
            sum += bin;

            if (src_bin == std::max(start_bin, 0)) {
              bin += left_outliers;
            }
            if (src_bin ==
                std::min(start_bin + nbins_selected - 1, nbins - 1)) {
              bin += right_outliers;
            }
            if (bin > 0) {
              non_zero_length += fraction;
            }
          }
        } // src_bin

        for (src_bin = src_bin_begin; src_bin < src_bin_end; ++src_bin) {
          if (src_bin >= 0 && src_bin < nbins) {
            uint64_t bin = bins[src_bin];
            double fraction = 1;
            if (src_bin == src_bin_begin && src_bin == src_bin_end - 1) {
              fraction = src_bin_end_not_rounded - src_bin_begin_not_rounded;
            } else if (src_bin == src_bin_begin) {
              fraction = (src_bin_begin + 1) - src_bin_begin_not_rounded;
            } else if (src_bin == src_bin_end - 1) {
              fraction = src_bin_end_not_rounded - (src_bin_end - 1);
            }

            if (src_bin == std::max(start_bin, 0)) {
              bin += left_outliers;
            }
            if (src_bin ==
                std::min(start_bin + nbins_selected - 1, nbins - 1)) {
              bin += right_outliers;
            }
            bin *= fraction;
            if (bin > 0) {
              double p = (double)bin / total_sum;
              double q = sum * fraction / non_zero_length / total_sum;
              kl += p * log(p / q);
            }
          }
        } // src_bin
      } // dst_bin

      assert(kl >= 0);
      if (kl < kl_min) {
        kl_min = kl;
        best_start_bin = start_bin;
      }
    } // for each start_bin

    best_start_bins[nbins_selected] = {best_start_bin, kl_min};
  } // for each nbins_selected

  double kl_min = numeric_limits<double>::max();
  int best_nbins_selected = dst_nbins, best_start_bin = 0;
  for (int nbins_selected = 1; nbins_selected <= nbins; ++nbins_selected) {
    double kl = best_start_bins[nbins_selected].second;
    if (kl < kl_min) {
      kl_min = kl;
      best_start_bin = best_start_bins[nbins_selected].first;
      best_nbins_selected = nbins_selected;
    }
  }

  double selected_sum = 0;
  int i_begin = std::max(0, best_start_bin);
  int i_end = std::min(nbins, best_start_bin + best_nbins_selected);
  for (int i = i_begin; i < i_end; ++i) {
    selected_sum += bins[i];
  }
  VLOG(2) << "best quantization range covers "
          << (double)selected_sum / total_sum * 100 << " %%";

  VLOG(2) << "best start_bin " << best_start_bin << " nbins_selected "
          << best_nbins_selected;

  min = hist.Min() + bin_width * (best_start_bin + 0.5);
  max = hist.Min() + bin_width * (best_start_bin + best_nbins_selected + 0.5);

  QuantizationFactory* qfactory = QuantizationFactory::GetDefaultInstance();
  return qfactory->ChooseQuantizationParams(min, max);
} // ChooseQuantizationParams

} // namespace dnnlowp
