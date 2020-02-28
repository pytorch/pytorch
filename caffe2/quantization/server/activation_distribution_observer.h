#pragma once

#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"
#include "caffe2/quantization/server/dnnlowp.h"
#include "caffe2/quantization/server/dynamic_histogram.h"

#include <memory>
#include <set>
#include <vector>

namespace caffe2 {

class OutputMinMaxObserver final : public ObserverBase<OperatorBase> {
 public:
  explicit OutputMinMaxObserver(OperatorBase* op);
  ~OutputMinMaxObserver();

  struct TensorInfo {
    explicit TensorInfo(const std::string& name)
        : min(std::numeric_limits<float>::max()),
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
  std::shared_ptr<OperatorInfo> GetInfo() {
    return info_;
  }

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
      NetBase* subject,
      const std::string& out_file_name,
      int dump_freq = -1,
      string delimiter = " ");
  ~OutputMinMaxNetObserver();

 private:
  void Stop() override;
  void DumpAndReset_(
      const std::string& out_file_name,
      bool print_total_min_max = false);

  int dump_freq_, cnt_;
  const std::string out_file_name_;
  std::string delimiter_;
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

  explicit HistogramObserver(OperatorBase* op, std::shared_ptr<Info> info);

 private:
  void Stop() override;

  std::shared_ptr<Info> info_;
  bool warning_printed_ = false;
}; // class HistogramObserver

/**
 * Given min/max, collect histogram of the max value of each column of tensor
 */
class OutputColumnMaxHistogramObserver final
    : public ObserverBase<OperatorBase> {
 public:
  explicit OutputColumnMaxHistogramObserver(
      OperatorBase* op,
      const std::string& col_max_blob_name,
      int nbins,
      std::shared_ptr<HistogramObserver::Info> info);

 private:
  void Stop() override;

  std::string col_max_blob_name_;
  int nbins_;
  std::shared_ptr<HistogramObserver::Info> info_;
  bool warning_printed_ = false;
  int col_max_blob_idx_ = -1;
  int num_columns_ = -1;
}; // class OutputColumnMaxHistogramObserver

class HistogramNetObserver final : public NetObserver {
 public:
  /**
   * @params mul_nets true if we expect multiple nets with the same name so
   *                  we include extra information in the file name to
   *                  distinghuish them
   * @params dump_freq if not -1 we dump histogram every dump_freq invocation
   *                   of the net
   */
  explicit HistogramNetObserver(
      NetBase* subject,
      const std::string& out_file_name,
      int nbins,
      int dump_freq = -1,
      bool mul_nets = false,
      string op_filter = "",
      string delimiter = " ");
  ~HistogramNetObserver();
  void DumpHistogramFile() {
    DumpAndReset_(out_file_name_, false);
  }

 private:
  void Stop() override;
  void DumpAndReset_(
      const std::string& out_file_name,
      bool print_total_min_max = false);

  int dump_freq_, cnt_;

  /** If multiple nets exist and are attached with the observers, the histogram
   * files for the nets will be appended with netbase addresses.
   */
  bool mul_nets_;
  string net_name_;
  string op_filter_;
  string delimiter_;
  const std::string out_file_name_;
  std::vector<std::shared_ptr<HistogramObserver::Info>> hist_infos_;
};

class OutputColumnMaxHistogramNetObserver final : public NetObserver {
 public:
  explicit OutputColumnMaxHistogramNetObserver(
      NetBase* subject,
      const std::string& out_file_name,
      const std::vector<std::string>& observe_column_max_for_blobs,
      int nbins,
      int dump_freq = -1,
      bool mul_nets = false,
      string delimiter = " ");
  ~OutputColumnMaxHistogramNetObserver();

 private:
  void Stop() override;
  void DumpAndReset_(
      const std::string& out_file_name,
      bool print_total_min_max = false);
  int dump_freq_, cnt_;
  bool mul_nets_;
  const std::string out_file_name_;
  std::string delimiter_;
  std::unordered_set<std::string> col_max_blob_names_;

  // {op_idx: {output_index: col_hists}}
  std::unordered_map<
      int,
      std::unordered_map<int, std::shared_ptr<HistogramObserver::Info>>>
      hist_infos_;
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

#ifdef _MSC_VER
struct tm* localtime_r(time_t* _clock, struct tm* _result) {
  struct tm* candidate_result = localtime(_clock);
  if (candidate_result) {
    *(_result) = *candidate_result;
  }
  return candidate_result;
}
#endif

} // namespace caffe2
