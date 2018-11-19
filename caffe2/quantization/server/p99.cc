#include "l2_minimization.h"
#include "caffe2/core/logging.h"

#include <cassert>
#include <cmath>

namespace dnnlowp {

TensorQuantizationParams P99::ChooseQuantizationParams(
    const Histogram& hist,
    bool preserve_sparsity,
    int /*precision*/) {
  assert(preserve_sparsity); // only support preserve_sparsity

  const std::vector<uint64_t> bins = *hist.GetHistogram();
  int nbins = bins.size();
  float min = hist.Min(), max = hist.Max();
  assert(min <= 0.f);
  assert(max >= 0.f);
  float bin_width = (max - min)/nbins;
  int zero_bin = ceil(-min/bin_width);

  int best_width = 0;
  double total_sum = 0;
  for (int i = 0; i < nbins; ++i) {
    total_sum += bins[i];
  }

  for (int width = 0; width < nbins; ++width) {
    int i_begin, i_end;
    if (min == 0) {
      i_begin = 0;
      i_end = width - 1;
    }
    else {
      i_begin = std::max(0, zero_bin - width);
      i_end = std::min(nbins - 1, zero_bin + width);
    }

    double selected_sum = 0;
    for (int i = i_begin; i <= i_end; ++i) {
      selected_sum += bins[i];
    }

    if (selected_sum / total_sum >= 0.99) {
      best_width = width;
      break;
    }
  }

  if (min == 0) {
    min = hist.Min();
    max = hist.Min() + bin_width * best_width;
  }
  else {
    min = hist.Min() + bin_width * (zero_bin - best_width);
    max = hist.Min() + bin_width * (zero_bin + best_width + 1);
  }

  QuantizationFactory *qfactory = QuantizationFactory::GetDefaultInstance();
  return qfactory->ChooseQuantizationParams(min, max);
} // ChooseQuantizationParams

} // namespace dnnlowp
