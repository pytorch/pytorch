#ifndef CAFFE2_ACTIVATION_DISTRIBUTION_OBSERVER_H
#define CAFFE2_ACTIVATION_DISTRIBUTION_OBSERVER_H

#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"
#include "caffe2/quantization/server/dnnlowp.h"
#include "caffe2/quantization/server/dynamic_histogram.h"

#include <vector>
#include <set>
#include <memory>

namespace caffe2 {

class OutputMinMaxObserver final : public ObserverBase<OperatorBase> {
 public:
  explicit OutputMinMaxObserver(OperatorBase *op);
  ~OutputMinMaxObserver();

  struct TensorInfo {
    TensorInfo(const std::string& name) :
      min(std::numeric_limits<float>::max()),
      max(std::numeric_limits<float>::lowest()),
      total_min(std::numeric_limits<float>::max()),
      total_max(std::numeric_limits<float>::lowest()),
      name(name) {}

    void Update(float cur_min, float cur_max) {
      min = std::min(min, cur_min);
      max = std::max(max, cur_max);
      total_min = std::min(total_min, cur_min);
      total_max = std::max(total_max, cur_max);
    }

    float min, max;
    float total_min, total_max;
    std::string name;
  };

  struct OperatorInfo {
    std::vector<TensorInfo> tensor_infos;
    std::string type;
  };

  // OutputMinMaxObserver is assumed to be used together with
  // OutputMinMaxNetObserver and the information shared via shared_ptr to be
  // prepared for the case when OutputMinMaxObserver is destroyed before
  // OutputMinMaxNetObserver
  std::shared_ptr<OperatorInfo> GetInfo() { return info_; }

 private:
  void Stop() override;

  std::shared_ptr<OperatorInfo> info_;
  bool warning_printed_ = false;
}; // class OutputMinMaxObserver

class OutputMinMaxNetObserver final : public NetObserver {
 public:
  /// @params dump_freq Print out only once in destructor if -1.
  //                    Otherwise, print out every dum_freq invocations
  explicit OutputMinMaxNetObserver(
    NetBase* subject, const std::string& out_file_name, int dump_freq = -1);
  ~OutputMinMaxNetObserver();

 private:
  void Stop() override;
  void DumpAndReset_(
    const std::string& out_file_name, bool print_total_min_max = false);

  int dump_freq_, cnt_;
  const std::string out_file_name_;
  std::vector<std::shared_ptr<OutputMinMaxObserver::OperatorInfo>>
    min_max_infos_;
};

/**
 * Given min/max, collect histogram
 */
class HistogramObserver final : public ObserverBase<OperatorBase> {
 public:
  struct Info {
    std::vector<dnnlowp::DynamicHistogram> histograms;
    std::vector<dnnlowp::DynamicHistogram> total_histograms;
    OutputMinMaxObserver::OperatorInfo min_max_info;
  };

  explicit HistogramObserver(OperatorBase *op, std::shared_ptr<Info> info);

 private:
  void Stop() override;

  std::shared_ptr<Info> info_;
  bool warning_printed_ = false;
}; // class HistogramObserver

class HistogramNetObserver final : public NetObserver {
 public:
  explicit HistogramNetObserver(
    NetBase* subject, const std::string& out_file_name, int nbins,
    int dump_freq = -1);
  ~HistogramNetObserver();

 private:
  void Stop() override;
  void DumpAndReset_(
    const std::string& out_file_name, bool print_total_min_max = false);

  int dump_freq_, cnt_;
  const std::string out_file_name_;
  std::vector<std::shared_ptr<HistogramObserver::Info>> hist_infos_;
};

/**
 * Set quantization parameters of operators based on min/max
 * collected from OutputMinMaxObserver
 */
class RegisterQuantizationParamsNetObserver final : public NetObserver {
 public:
  explicit RegisterQuantizationParamsNetObserver(
      NetBase* subject,
      const std::string& min_max_file_name,
      bool is_weight = false,
      const std::string& qparams_output_file_name = "");
};

/**
 * Set quantization parameters of operators based on min/max
 * collected from OutputMinMaxObserver
 */
class RegisterQuantizationParamsWithHistogramNetObserver final
  : public NetObserver {
 public:
  explicit RegisterQuantizationParamsWithHistogramNetObserver(
      NetBase* subject,
      const std::string& histogram_file_name,
      bool is_weight = false,
      const std::string& qparams_output_file_name = "");
};

} // namespace caffe2

#endif // CAFFE2_ACTIVATION_DISTRIBUTION_OBSERVER_H
