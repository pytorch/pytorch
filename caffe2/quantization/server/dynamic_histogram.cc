#include "dynamic_histogram.h"
#include "dnnlowp_op.h"

#include <cassert>
#include <cmath>
#include <limits>

namespace dnnlowp {

using namespace std;

void Histogram::Add(float f, uint64_t cnt) {
  int nbins = histogram_.size();
  float bin_width = (max_ - min_) / nbins;
  int bin = bin_width == 0
      ? 0
      : std::min(static_cast<int>((f - min_) / bin_width), nbins - 1);
  bin = std::max(0, bin);
  assert(bin >= 0);
  histogram_[bin] += cnt;
}

void Histogram::Add(const float* f, int len) {
  int nbins = histogram_.size();
  float bin_width = (max_ - min_) / nbins;

  if (bin_width > 0.0) {
    uint64_t* my_histogram = nullptr;
    my_histogram = histogram_.data();

    for (auto i = 0; i < len; ++i) {
      int bin =
          std::min(static_cast<int>((f[i] - min_) / bin_width), nbins - 1);
      bin = std::max(0, bin);
      ++my_histogram[bin];
    }
  } else {
    histogram_[0] += len;
  }
}

void RemapHistograms(Histogram& src_hist, Histogram& dst_hist) {
  auto src_bins = *(src_hist.GetHistogram());
  float src_bin_width = (src_hist.Max() - src_hist.Min()) / src_bins.size();
  float dst_bin_width =
      (dst_hist.Max() - dst_hist.Min()) / dst_hist.GetHistogram()->size();
  for (int i = 0; i < src_bins.size(); ++i) {
    if (src_bins[i] == 0) {
      continue;
    }
    float src_bin_begin = src_hist.Min() + src_bin_width * i;
    float src_bin_end = src_bin_begin + src_bin_width;

    // dst_bin corresponds to the beginning of the src_bin
    // dst_bin2 corresponds to the end of the src_bin
    int dst_bin = dst_bin_width == 0
        ? 0
        : (src_bin_begin - dst_hist.Min()) / dst_bin_width;
    float dst_bin_begin = dst_hist.Min() + dst_bin_width * dst_bin;
    float dst_bin_end = dst_bin_begin + dst_bin_width;
    int dst_bin2 =
        dst_bin_width == 0 ? 0 : (src_bin_end - dst_hist.Min()) / dst_bin_width;
    // 1 src_bin is mapped to at most 2 dst bin
    assert(dst_bin2 <= dst_bin + 2);

    // dst_bin_cnt is the count from src_bin that should go to dst_bin
    // The remainder should go to dst_bin2
    // rint is the fastest way to round
    // (https://stackoverflow.com/questions/485525/round-for-float-in-c/5849630)
    uint64_t dst_bin_cnt = (src_bin_width == 0 || dst_bin_width == 0)
        ? src_bins[i]
        : std::min(
              static_cast<uint64_t>(rint(
                  (dst_bin_end - src_bin_begin) / src_bin_width * src_bins[i])),
              src_bins[i]);

    dst_hist.Add(dst_bin_begin + dst_bin_width / 2, dst_bin_cnt);
    if (dst_bin_cnt < src_bins[i]) {
      dst_hist.Add(dst_bin_end + dst_bin_width / 2, src_bins[i] - dst_bin_cnt);
    }
  }
}

static const int OVER_BINNING_FACTOR = 4;

DynamicHistogram::DynamicHistogram(int nbins)
    : nbins_(nbins),
      min_(numeric_limits<float>::max()),
      max_(numeric_limits<float>::lowest()) {
  assert(nbins_ > 0);
}

void DynamicHistogram::Add(float f) {
  min_ = std::min(min_, f);
  max_ = std::max(max_, f);

  if (histogram_ == nullptr) {
    histogram_ =
        std::make_unique<Histogram>(nbins_ * OVER_BINNING_FACTOR, min_, max_);
    histogram_->Add(f);
    return;
  }
  Histogram curr_hist = *histogram_;
  float new_min = curr_hist.Min(), new_max = curr_hist.Max();
  if (f < curr_hist.Min() || f > curr_hist.Max()) {
    float old_spread = curr_hist.Max() - curr_hist.Min();
    if (f < curr_hist.Min()) {
      if (old_spread == 0) {
        new_min = f;
      } else {
        new_min = curr_hist.Min() -
            ceil((curr_hist.Min() - f) / old_spread) * old_spread;
      }
    } else {
      if (old_spread == 0) {
        new_max = f;
      } else {
        new_max = curr_hist.Max() +
            ceil((f - curr_hist.Max()) / old_spread) * old_spread;
      }
    }
    new_min = std::max(numeric_limits<float>::lowest(), new_min);
    new_max = std::min(numeric_limits<float>::max(), new_max);
    histogram_.reset(
        new Histogram(curr_hist.GetHistogram()->size(), new_min, new_max));
    RemapHistograms(curr_hist, *histogram_);
  }
  histogram_->Add(f);
}

void DynamicHistogram::Add(const float* f, int len) {
  float minimum = min_, maximum = max_;
  for (int i = 0; i < len; ++i) {
    minimum = std::min(f[i], minimum);
    maximum = std::max(f[i], maximum);
  }
  min_ = std::max(numeric_limits<float>::lowest(), minimum);
  max_ = std::min(numeric_limits<float>::max(), maximum);

  if (histogram_ == nullptr) {
    histogram_ =
        std::make_unique<Histogram>(nbins_ * OVER_BINNING_FACTOR, min_, max_);
    histogram_->Add(f, len);
    return;
  }
  Histogram curr_hist = *histogram_;
  float new_min = curr_hist.Min(), new_max = curr_hist.Max();
  if (min_ < curr_hist.Min() || max_ > curr_hist.Max()) {
    float old_spread = curr_hist.Max() - curr_hist.Min();
    if (min_ < curr_hist.Min()) {
      if (old_spread == 0.0f) {
        new_min = min_;
      } else {
        new_min = curr_hist.Min() -
            ceil((curr_hist.Min() - min_) / old_spread) * old_spread;
      }
    }
    if (max_ > curr_hist.Max()) {
      old_spread = curr_hist.Max() - new_min;
      if (old_spread == 0.0f) {
        new_max = max_;
      } else {
        new_max = curr_hist.Max() +
            ceil((max_ - curr_hist.Max()) / old_spread) * old_spread;
      }
    }
    new_min = std::max(numeric_limits<float>::lowest(), new_min);
    new_max = std::min(numeric_limits<float>::max(), new_max);
    histogram_.reset(
        new Histogram(curr_hist.GetHistogram()->size(), new_min, new_max));
    RemapHistograms(curr_hist, *histogram_);
  }

  histogram_->Add(f, len);
}

const Histogram* DynamicHistogram::Finalize() {
  if (final_histogram_.get()) {
    return final_histogram_.get();
  }

  final_histogram_.reset(new Histogram(nbins_, min_, max_));
  if (histogram_.get()) {
    RemapHistograms(*histogram_, *final_histogram_);
  }

  return final_histogram_.get();
}

} // namespace dnnlowp
