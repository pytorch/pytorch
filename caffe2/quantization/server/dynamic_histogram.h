#pragma once

#include <memory>
#include <vector>

namespace dnnlowp {

/**
 * bin_width = (max - min)/nbins
 * ith bin (zero-based indexing) contains [i*bin_width, (i+1)*bin_width)
 * with an exception that (nbins - 1)th bin contains
 * [(nbins-1)*bin_width, nbins*bin_width]
 *
 */
class Histogram {
 public:
  Histogram(int nbins, float min, float max)
      : min_(min), max_(max), histogram_(nbins) {}
  Histogram(float min, float max, const std::vector<uint64_t>& bins)
      : min_(min), max_(max), histogram_(bins) {}

  void Add(float f, uint64_t cnt = 1);
  /**
   * This version collects histogram with single thread
   */
  void Add(const float* f, int len);

  float Min() const {
    return min_;
  }
  float Max() const {
    return max_;
  }

  const std::vector<uint64_t>* GetHistogram() const {
    return &histogram_;
  }

 private:
  float min_, max_;
  std::vector<uint64_t> histogram_;
};

/// An equi-width histogram where the spread of bins change over time when
/// we see new min or max values.
class DynamicHistogram {
 public:
  DynamicHistogram(int nbins);

  void Add(float f);
  void Add(const float* f, int len);

  /// Indicate we're not dynamically adjusting histogram bins any more and
  /// return the current static histogram.
  const Histogram* Finalize();

 private:
  /// Dynamic histogram is implemented by the series of static histograms
  /// and expands from the old histogram to new histogram when
  /// we see a new extremum.
  /// An invariant: the beginning of the first bin of histograms_[i] exactly
  /// matches with the beginning of a bin in histograms_[i+1]. The end of the
  /// last bin of histograms_[i] exactly matches with the end of a bin in
  /// histograms_[i+1].
  std::unique_ptr<Histogram> histogram_;
  int nbins_;
  float min_, max_;

  std::unique_ptr<Histogram> final_histogram_;
}; // class DynamicHistogram

} // namespace dnnlowp
