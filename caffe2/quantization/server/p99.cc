#include "caffe2/core/logging.h"
#include "l2_minimization.h"

#include <cassert>
#include <cmath>

namespace dnnlowp {

TensorQuantizationParams P99::ChooseQuantizationParams(
    const Histogram& hist,
    bool preserve_sparsity,
    int precision) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  float min, max;
  std::vector<float> bins_f(
      dnnlowp::adjust_hist_to_include_zero(hist, &min, &max));
  int nbins = bins_f.size();
  CAFFE_ENFORCE(min <= 0.f);
  CAFFE_ENFORCE(max >= 0.f);
  float org_max = max;
  float org_min = min;
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  float bin_width = (max - min) / nbins;

  double total_sum = 0;
  for (int i = 0; i < nbins; ++i) {
    total_sum += bins_f[i];
  }
  double sum = 0;
  std::vector<double> CDF(nbins, 0.f);
  for (int i = 0; i < nbins; ++i) {
    sum += bins_f[i];
    CDF[i] = (double)sum / total_sum;
  }
  CAFFE_ENFORCE(threshold_ > 0.5 && threshold_ < 1);
  double left_quantile = (1.0f - threshold_) / 2.0f;
  double right_quantile = 1.0f - left_quantile;
  int i_begin = 0;
  int i_end = nbins - 2;
  bool finished = false;
  while (i_begin <= i_end && !finished) {
    finished = true;
    if (CDF[i_begin] < left_quantile) {
      i_begin++;
      finished = false;
    }
    if (CDF[i_end] > right_quantile) {
      finished = false;
      i_end--;
    }
  }
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  min = i_begin * bin_width + org_min;
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  max = (i_end + 2) * bin_width + org_min;

  VLOG(2) << "Org min " << org_min << " org max " << org_max << " found min "
          << min << " max " << max;

  QuantizationFactory* qfactory = QuantizationFactory::GetDefaultInstance();
  return qfactory->ChooseQuantizationParams(
      min, max, precision, preserve_sparsity);
} // ChooseQuantizationParams

} // namespace dnnlowp
