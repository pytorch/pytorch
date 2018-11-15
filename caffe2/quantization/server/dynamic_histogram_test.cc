#include <algorithm>
#include <array>
#include <random>
#include <sstream>

#include <gtest/gtest.h>
#include "caffe2/core/logging.h"

#include "dynamic_histogram.h"

using namespace std;
using namespace dnnlowp;

TEST(DynamicHistogram, HistSimilar) {
  default_random_engine generator;
  normal_distribution<float> distribution;

  constexpr int n = 65536;
  array<float, n> data; //make_array<float>(n);

  for (int i = 0; i < n; ++i) {
    data[i] = distribution(generator);
  }

  // Construct static and dynamic histogram and compare.
  float minimum = *min_element(data.begin(), data.end());
  float maximum = *max_element(data.begin(), data.end());

  int nbins = 64;
  Histogram static_hist(nbins, minimum, maximum);
  vector<unique_ptr<DynamicHistogram>> dynamic_hist(3);
  for (auto i = 0; i < dynamic_hist.size(); ++i) {
    dynamic_hist[i].reset(new DynamicHistogram(nbins));
  }

  static_hist.Add(data.data(), n);
  dynamic_hist[0]->Add(data.data(), n);
  for (int i = 0; i < n; ++i) {
    dynamic_hist[1]->Add(data[i]);
  }
  for (int i = 0; i < 64; ++i) {
    dynamic_hist[2]->Add(data.data() + n / 64 * i, n / 64);
  }

  stringstream ss;
  static_hist.Finalize();
  for (int i = 0; i < nbins; ++i) {
    ss << (*static_hist.GetHistogram())[i] << " ";
  }
  LOG(INFO) << "static: " << ss.str();

  vector<float> errors(dynamic_hist.size());
  for (auto i = 0; i < dynamic_hist.size(); ++i) {
    ss.str("");
    const Histogram *dynamic_hist_result = dynamic_hist[i]->Finalize();
    for (auto j = 0; j < nbins; ++j) {
      ss << (*dynamic_hist_result->GetHistogram())[j] << " ";
    }
    LOG(INFO) << "dynamic " << i << " : " << ss.str();

    // Compute the normalized squared error between the two histograms
    float error = 0.0;
    for (int j = 0; j < nbins; ++j) {
      float e = (float)(*static_hist.GetHistogram())[j] -
                (*dynamic_hist_result->GetHistogram())[j];
      error += e * e;
    }
    error /= n;
    LOG(INFO) << "error: " << error << endl;
    errors[i] = error;
  }

  for (auto i = 0; i < dynamic_hist.size(); ++i) {
    EXPECT_TRUE(errors[i] < 0.3);
  }
}
