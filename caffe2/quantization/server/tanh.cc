#include "tanh.h"
#include <cassert>
#include "caffe2/core/logging.h"

namespace dnnlowp {

static double GetSaturationRegionBegin_(double max_abs_err) {
  // smallest x_s s.t. 1 - tanh(x_s) < max_abs_err_ and is an integer
  double x_s = atanh(1 - max_abs_err);
  if (x_s < 1) {
    return 1 / floor(1 / x_s);
  } else {
    return ceil(x_s);
  }
}

static int GetPassRegionEnd_(
    TensorQuantizationParams in_qparams,
    TensorQuantizationParams out_qparams,
    double max_abs_err,
    int num_in_bits) {
  return 0;

  // largest x s.t. |tanh(x) - x| < max_abs_err_
  int in_pos_qmax = (1 << (num_in_bits - 1)) - 1;

  float scale_multiplier = in_qparams.scale / out_qparams.scale;
  int log2_scale_multiplier = nearbyint(log2(scale_multiplier));

  int x_q;
  for (x_q = 0; x_q < in_pos_qmax; ++x_q) {
    int y_q;
    if (log2_scale_multiplier < 0) {
      y_q = x_q >> (-log2_scale_multiplier);
    } else {
      y_q = x_q << (log2_scale_multiplier);
    }
    float y = y_q * out_qparams.scale;

    float x_min = std::max((x_q - 0.5f) * in_qparams.scale, 0.f);
    float x_max = (x_q + 0.5f) * in_qparams.scale;
    if (fabs(tanh(x_max) - y) > max_abs_err ||
        fabs(tanh(x_min) - y) > max_abs_err) {
      break;
    }
  }
  return x_q - 1;
}

template <typename T>
Tanh<T>::Tanh(double max_abs_err) : max_abs_err_(max_abs_err) {
  // Choose saturation region
  double x_sq = GetSaturationRegionBegin_(max_abs_err);

  // Choose input/output quantization parameters
  in_qparams_.scale = x_sq / ((1 << (num_in_bits_ - 1)) - 1);
  in_qparams_.zero_point = 1 << (num_in_bits_ - 1);
  in_qparams_.precision = num_in_bits_;
  // -x_sq is mapped to -127, 0 is mapped to 0, x_sq is mapped to 127

  out_qparams_.scale = 1. / ((1 << (num_out_bits_ - 1)) - 1);
  out_qparams_.zero_point = 1 << (num_out_bits_ - 1);
  out_qparams_.precision = num_out_bits_;
  // -1 is mapped to -127, 0 is mapped to 0, x_sq is mapped to 127

  // Choose pass region
  x_pq_index_ =
      GetPassRegionEnd_(in_qparams_, out_qparams_, max_abs_err, num_in_bits_);

  int in_pos_qmax = (1 << (num_in_bits_ - 1)) - 1;
  processing_region_lut_.resize(in_pos_qmax - x_pq_index_ + 2);

  int i;
  for (i = x_pq_index_; i < in_pos_qmax; ++i) {
    double y_begin = tanh((i - 0.5) * in_qparams_.scale);
    double y_end = tanh((i + 0.5) * in_qparams_.scale);

    int y_avg_q = nearbyint((y_begin + y_end) / 2 / out_qparams_.scale);
    assert(y_avg_q * out_qparams_.scale - y_begin < max_abs_err);
    assert(y_end - y_avg_q * out_qparams_.scale < max_abs_err);
    assert(y_avg_q >= 0);
    assert(y_avg_q < (1 << (num_out_bits_ - 1)));
    processing_region_lut_[i - x_pq_index_] = y_avg_q;
#ifdef PRINT_TANH_TABLE
    LOG(INFO) << i << " " << y_avg_q;
#endif
  }
  // saturation region: for 8-bit, -128 and -127 map to -1, and 127 map to 1
  processing_region_lut_[i - x_pq_index_] = (1 << (num_out_bits_ - 1)) - 1;
#ifdef PRINT_TANH_TABLE
  LOG(INFO) << i << " " << processing_region_lut_[i - x_pq_index_];
#endif
  processing_region_lut_[i - x_pq_index_ + 1] = (1 << (num_out_bits_ - 1)) - 1;
#ifdef PRINT_TANH_TABLE
  LOG(INFO) << i + 1 << " " << processing_region_lut_[i - x_pq_index_ + 1];
#endif
}

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T>
T Tanh<T>::Compute(T x) const {
  int32_t x_adjusted = x - in_qparams_.zero_point;
  int32_t x_sgn = sgn(x_adjusted), x_mag = std::abs(x_adjusted);
  int32_t y;

  if (x_mag < x_pq_index_) {
    // pass region
    float scale_multiplier = in_qparams_.scale / out_qparams_.scale;
    int log2_scale_multiplier = nearbyint(log2(scale_multiplier));
    if (log2_scale_multiplier < 0) {
      y = x_sgn * (x_mag >> (-log2_scale_multiplier));
    } else {
      y = x_sgn * (x_mag << log2_scale_multiplier);
    }
  } else {
    // processing and saturation region
    y = x_sgn * processing_region_lut_[x_mag - x_pq_index_];
  }

  assert(y + out_qparams_.zero_point <= std::numeric_limits<T>::max());

  // assuming output is unsigned
  assert(y + out_qparams_.zero_point >= 0);
  assert(y + out_qparams_.zero_point < (1 << num_out_bits_));

  return y + out_qparams_.zero_point;
}

template class Tanh<uint8_t>;
template class Tanh<uint16_t>;
template class Tanh<int32_t>;

} // namespace dnnlowp
