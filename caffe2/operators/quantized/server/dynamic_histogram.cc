#include "dynamic_histogram.h"
#include "dnnlowp_op.h"

#include <cassert>
#include <limits>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dnnlowp {

using namespace std;

void Histogram::Add(float f, uint64_t cnt) {
  int nbins = histogram_.size();
  float bin_width = (max_ - min_) / nbins;
  int bin =
    bin_width == 0
    ? 0 : std::min(static_cast<int>((f - min_)/bin_width), nbins - 1);
  bin = std::max(0, bin);
  assert(bin >= 0);
  histogram_[bin] += cnt;
}

void Histogram::Add(const float* f, int len) {
  int nbins = histogram_.size();
  float bin_width = (max_ - min_) / nbins;

  if (bin_width > 0.0) {
    assert(per_thread_histogram_.size() % nbins == 0);

    // Check if dnnlowp_get_max_threads has been reduced, and if so reduce
    // per-thread histogram and clear them.
    int old_nthreads = per_thread_histogram_.size() / nbins + 1;
    if (caffe2::dnnlowp_get_max_threads() < old_nthreads) {
      Finalize();
    }

    per_thread_histogram_.resize((caffe2::dnnlowp_get_max_threads() - 1) * nbins);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = caffe2::dnnlowp_get_thread_num();

      uint64_t* my_histogram = nullptr;
      if (tid == 0) {
        my_histogram = histogram_.data();
      } else {
        my_histogram = per_thread_histogram_.data() + (tid - 1) * nbins;
      }

#ifdef _OPENMP
#pragma omp for
#endif
      for (auto i = 0; i < len; ++i) {
        int bin =
            std::min(static_cast<int>((f[i] - min_) / bin_width), nbins - 1);
        bin = std::max(0, bin);
        ++my_histogram[bin];
      }
    } // omp parallel
  } else {
    histogram_[0] += len;
  }
}

void Histogram::Finalize() {
  int nbins = histogram_.size();
  assert(per_thread_histogram_.size() % nbins == 0);
  int nthreads = per_thread_histogram_.size() / nbins + 1;

  if (nthreads > 1) {
    for (int bin = 0; bin < nbins; ++bin) {
      for (int i = 1; i < nthreads; ++i) {
        histogram_[bin] += per_thread_histogram_[(i - 1) * nbins + bin];
      }
    }
  }

  per_thread_histogram_.clear();
}

static const int OVER_BINNING_FACTOR = 4;

DynamicHistogram::DynamicHistogram(int nbins)
  : nbins_(nbins),
    min_(numeric_limits<float>::max()), max_(numeric_limits<float>::lowest()) {
  assert(nbins_ > 0);
}

void DynamicHistogram::Add(float f) {
  min_ = std::min(min_, f);
  max_ = std::max(max_, f);

  if (histograms_.empty()) {
    histograms_.emplace_back(nbins_ * OVER_BINNING_FACTOR, f, f);
  } else {
    Histogram& curr_hist = histograms_.back();
    if (f < curr_hist.Min() || f > curr_hist.Max()) {
      float old_spread = curr_hist.Max() - curr_hist.Min();
      if (f < curr_hist.Min()) {
        float new_min;
        if (old_spread == 0) {
          new_min = f;
        } else {
          new_min = curr_hist.Min() -
              ceil((curr_hist.Min() - f) / old_spread) * old_spread;
        }
        histograms_.emplace_back(
          curr_hist.GetHistogram()->size(), new_min, curr_hist.Max());
      } else {
        float new_max;
        if (old_spread == 0) {
          new_max = f;
        } else {
          new_max = curr_hist.Max() +
              ceil((f - curr_hist.Max()) / old_spread) * old_spread;
        }
        histograms_.emplace_back(
          curr_hist.GetHistogram()->size(), curr_hist.Min(), new_max);
      }
    }
  }

  Histogram& new_hist = histograms_.back();
  new_hist.Add(f);
}

void DynamicHistogram::Add(const float* f, int len) {
  float minimum = min_, maximum = max_;
#ifdef _OPENMP
#pragma omp parallel for reduction(min : minimum) reduction(max : maximum)
#endif
  for (int i = 0; i < len; ++i) {
    minimum = std::min(f[i], minimum);
    maximum = std::max(f[i], maximum);
  }
  min_ = minimum;
  max_ = maximum;

  if (histograms_.empty()) {
    histograms_.emplace_back(nbins_ * OVER_BINNING_FACTOR, min_, max_);
  } else {
    Histogram& curr_hist = histograms_.back();
    if (min_ < curr_hist.Min() || max_ > curr_hist.Max()) {
      float old_spread = curr_hist.Max() - curr_hist.Min();
      float new_min = curr_hist.Min(), new_max = curr_hist.Max();
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
      histograms_.emplace_back(
          curr_hist.GetHistogram()->size(), new_min, new_max);
    }
  }

  Histogram& new_hist = histograms_.back();
  new_hist.Add(f, len);
}

const Histogram *DynamicHistogram::Finalize() {
  if (final_histogram_.get()) {
    return final_histogram_.get();
  }

  final_histogram_.reset(new Histogram(nbins_, min_, max_));
  float dst_bin_width = (max_ - min_) / nbins_;

  for (Histogram& hist : histograms_) {
    hist.Finalize();

    const std::vector<uint64_t>& bins = *hist.GetHistogram();
    float src_bin_width = (hist.Max() - hist.Min()) / bins.size();

    for (int i = 0; i < bins.size(); ++i) {
      if (bins[i] == 0) continue;
      float src_bin_begin = hist.Min() + src_bin_width * i;
      float src_bin_end = src_bin_begin + src_bin_width;
      // dst_bin corresponds to the beginning of the src_bin
      // dst_bin2 corresponds to the end of the src_bin
      int dst_bin =
          dst_bin_width == 0 ? 0 : (src_bin_begin - min_) / dst_bin_width;
      float dst_bin_begin = min_ + dst_bin_width * dst_bin;
      float dst_bin_end = dst_bin_begin + dst_bin_width;
      int dst_bin2 =
          dst_bin_width == 0 ? 0 : (src_bin_end - min_) / dst_bin_width;
      // 1 src_bin is mapped to at most 2 dst bin
      assert(dst_bin2 <= dst_bin + 2);

      // dst_bin_cnt is the count from src_bin that should go to dst_bin
      // The remainder should go to dst_bin2
      // TODO: This is only run at the beginning when there is only
      // one bin, we should optimize this with the unlikely compiler hints
      uint64_t dst_bin_cnt = src_bin_width == 0 ? 0 : std::min(
        (uint64_t)round(
          (dst_bin_end - src_bin_begin) / src_bin_width * bins[i]),
        bins[i]);

      final_histogram_->Add(dst_bin_begin + dst_bin_width / 2, dst_bin_cnt);
      if (dst_bin_cnt < bins[i]) {
        final_histogram_->Add(
            dst_bin_end + dst_bin_width / 2, bins[i] - dst_bin_cnt);
      }
    }
  } // for each histogram with different scales

  return final_histogram_.get();
}

} // namespace dnnlowp
