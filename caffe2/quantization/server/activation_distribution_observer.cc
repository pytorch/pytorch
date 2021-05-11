#include "caffe2/quantization/server/activation_distribution_observer.h"
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"
#include "caffe2/quantization/server/dnnlowp.h"

namespace caffe2 {

using namespace std;
using namespace dnnlowp;

OutputMinMaxObserver::OutputMinMaxObserver(OperatorBase* op)
    : ObserverBase<OperatorBase>(op), info_(make_shared<OperatorInfo>()) {
  for (int i = 0; i < op->OutputSize(); ++i) {
    info_->tensor_infos.emplace_back(op->debug_def().output(i));
    info_->type = op->debug_def().type();
  }
}

// A global table that collects min/max for each tensor name.
// Useful in case there are multiple copies of the same network.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static map<string, pair<float, float>> min_max_map_;

// NOLINTNEXTLINE(modernize-use-equals-default)
OutputMinMaxObserver::~OutputMinMaxObserver() {
  /*#pragma omp critical
    {
      for (int i = 0; i < info_->tensor_infos.size(); ++i) {
        LOG(INFO) <<
          this << " " << info_->type << " " << i << " " <<
          info_->tensor_infos[i].name << " " <<
          info_->tensor_infos[i].min << " " <<
          info_->tensor_infos[i].max << " " <<
          min_max_map_[info_->tensor_infos[i].name].first << " " <<
          min_max_map_[info_->tensor_infos[i].name].second << " ";
      }
    }*/
}

template <typename T>
void FindMinMax(const T* data, float* min, float* max, int len) {
  vector<float> temp(len);
  for (int i = 0; i < len; ++i) {
    temp[i] = data[i];
  }
  fbgemm::FindMinMax(temp.data(), min, max, len);
}

float* GetFloatTensorData(TensorCPU* tensor) {
  float* data = nullptr;
  vector<float> data_temp;
  if (tensor->IsType<float>()) {
    if (!tensor->data<float>()) {
      return nullptr;
    }
    data = tensor->template data<float>();
  } else if (tensor->IsType<int>()) {
    if (!tensor->data<int>()) {
      return nullptr;
    }
    const int* data_orig = tensor->data<int>();
    data_temp.resize(tensor->numel());
    for (int j = 0; j < tensor->numel(); ++j) {
      data_temp[j] = data_orig[j];
    }
    data = data_temp.data();
  } else if (tensor->IsType<long>()) {
    if (!tensor->data<long>()) {
      return nullptr;
    }
    const long* data_orig = tensor->data<long>();
    data_temp.resize(tensor->numel());
    for (int j = 0; j < tensor->numel(); ++j) {
      data_temp[j] = data_orig[j];
    }
    data = data_temp.data();
  } else {
    return nullptr;
  }
  return data;
}

template <>
void FindMinMax<float>(const float* data, float* min, float* max, int len) {
  fbgemm::FindMinMax(data, min, max, len);
}

void OutputMinMaxObserver::Stop() {
  for (int i = 0; i < subject_->OutputSize(); ++i) {
    if (!subject_->OutputIsTensorType(i, CPU)) {
      continue;
    }
    Tensor* tensor = subject_->template Output<Tensor>(i, CPU);
    if (!tensor || tensor->numel() == 0 || tensor->numel() == -1)
      continue;
    string out_name(subject_->debug_def().output(i));

    float min = numeric_limits<float>::lowest(),
          max = numeric_limits<float>::max();

    if (tensor->IsType<float>()) {
      if (!tensor->data<float>()) {
        continue;
      }
      FindMinMax(tensor->data<float>(), &min, &max, tensor->numel());
    } else if (tensor->IsType<int>()) {
      if (!tensor->data<int>()) {
        continue;
      }
      FindMinMax(tensor->data<int>(), &min, &max, tensor->numel());
    } else if (tensor->IsType<long>()) {
      if (!tensor->data<long>()) {
        continue;
      }
      FindMinMax(tensor->data<long>(), &min, &max, tensor->numel());
    } else {
      if (!warning_printed_) {
        LOG(INFO) << "Tensor " << out_name << " has unsupported type "
                  << tensor->meta().name() << " with size " << tensor->numel();
        warning_printed_ = true;
      }
      continue;
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    {
      if (min_max_map_.find(out_name) == min_max_map_.end()) {
        min_max_map_[out_name] = make_pair(
            numeric_limits<float>::max(), numeric_limits<float>::lowest());
      }

      info_->tensor_infos[i].Update(min, max);

      min_max_map_[out_name].first =
          std::min(min_max_map_[out_name].first, min);
      min_max_map_[out_name].second =
          std::max(min_max_map_[out_name].second, max);
      assert(min_max_map_[out_name].second >= min_max_map_[out_name].first);
      assert(min_max_map_[out_name].first < 1e38);

      VLOG(2) << this << " " << info_->type << " " << i << " " << out_name
              << " " << info_->tensor_infos[i].min << " "
              << info_->tensor_infos[i].max << " "
              << min_max_map_[out_name].first << " "
              << min_max_map_[out_name].second;
    }
  }

  return;
}

OutputMinMaxNetObserver::OutputMinMaxNetObserver(
    NetBase* subject,
    const string& out_file_name,
    int dump_freq,
    // NOLINTNEXTLINE(modernize-pass-by-value)
    string delimiter)
    : NetObserver(subject),
      dump_freq_(dump_freq),
      cnt_(0),
      out_file_name_(out_file_name),
      delimiter_(delimiter) {
  VLOG(2) << out_file_name;
  min_max_infos_.resize(subject->GetOperators().size());
  int i = 0;
  for (auto* op : subject->GetOperators()) {
    OutputMinMaxObserver* observer = new OutputMinMaxObserver(op);
    op->AttachObserver(std::unique_ptr<OutputMinMaxObserver>(observer));
    min_max_infos_[i] = observer->GetInfo();
    ++i;
  }
}

void OutputMinMaxNetObserver::DumpAndReset_(
    const std::string& out_file_name,
    bool print_total_min_max) {
  ofstream f(out_file_name);
  if (!f) {
    LOG(WARNING) << this << ": can't open " << out_file_name;
  }

  // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
  for (int op_index = 0; op_index < min_max_infos_.size(); ++op_index) {
    OutputMinMaxObserver::OperatorInfo* op_info =
        min_max_infos_[op_index].get();
    if (op_info) {
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (int i = 0; i < op_info->tensor_infos.size(); ++i) {
        const OutputMinMaxObserver::TensorInfo& tensor_info =
            op_info->tensor_infos[i];

        ostringstream ost;
        ost << op_index << delimiter_ << op_info->type << delimiter_ << i
            << delimiter_ << tensor_info.name << delimiter_;
        if (print_total_min_max) {
          ost << tensor_info.total_min << delimiter_ << tensor_info.total_max;
        } else {
          ost << tensor_info.min << delimiter_ << tensor_info.max;
        }

        LOG(INFO) << this << delimiter_ << ost.str();
        f << ost.str() << endl;

        op_info->tensor_infos[i].min = numeric_limits<float>::max();
        op_info->tensor_infos[i].max = numeric_limits<float>::lowest();
      }
    }
  }
  f.close();
}

// NOLINTNEXTLINE(bugprone-exception-escape)
OutputMinMaxNetObserver::~OutputMinMaxNetObserver() {
  DumpAndReset_(out_file_name_, true);

#ifdef _OPENMP
#pragma omp critical
#endif
  {
    ofstream f;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    time_t rawtime;
    time(&rawtime);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    struct tm timeinfo;
    localtime_r(&rawtime, &timeinfo);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
    char buffer[128] = {};
    strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H-%M-%S", &timeinfo);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
    char buffer2[256] = {};
    snprintf(buffer2, sizeof(buffer2), "global_%s.minmax", buffer);

    f.open(buffer2);
    int op_index = 0;
    // NOLINTNEXTLINE(performance-for-range-copy)
    for (auto key_value : min_max_map_) {
      ostringstream ost;
      assert(key_value.second.first <= key_value.second.second);
      assert(key_value.second.first < 1e38);
      ost << op_index << " 0 " << key_value.first << " "
          << key_value.second.first << " " << key_value.second.second;
      f << ost.str() << endl;

      ++op_index;
    }
    f.close();
  }
}

void OutputMinMaxNetObserver::Stop() {
  ++cnt_;
  if (dump_freq_ == -1 || (cnt_ % dump_freq_) != 0) {
    return;
  }

  ostringstream ost;
  size_t last_dot = out_file_name_.rfind('.');
  size_t last_slash = out_file_name_.rfind('/');
  if (last_dot != string::npos &&
      (last_slash == string::npos || last_slash < last_dot)) {
    ost << out_file_name_.substr(0, last_dot) << "_" << cnt_ / dump_freq_
        << out_file_name_.substr(last_dot);
  } else {
    ost << out_file_name_ << "_" << cnt_ / dump_freq_;
  }

  DumpAndReset_(ost.str());
  return;
}

// NOLINTNEXTLINE(modernize-pass-by-value)
HistogramObserver::HistogramObserver(OperatorBase* op, shared_ptr<Info> info)
    : ObserverBase<OperatorBase>(op), info_(info) {}

void HistogramObserver::Stop() {
  for (int i = 0; i < subject_->OutputSize(); ++i) {
    if (!subject_->OutputIsTensorType(i, CPU)) {
      continue;
    }
    Tensor* tensor = subject_->template Output<Tensor>(i, CPU);
    if (!tensor || tensor->numel() == 0 || tensor->numel() == -1) {
      continue;
    }

    string out_name(subject_->debug_def().output(i));

    const float* data = nullptr;
    vector<float> data_temp;

    if (tensor->IsType<float>()) {
      if (!tensor->data<float>()) {
        continue;
      }
      data = tensor->template data<float>();
    } else if (tensor->IsType<int>()) {
      if (!tensor->data<int>()) {
        continue;
      }
      const int* data_orig = tensor->data<int>();
      data_temp.resize(tensor->numel());
      for (int j = 0; j < tensor->numel(); ++j) {
        data_temp[j] = data_orig[j];
      }
      data = data_temp.data();
    } else if (tensor->IsType<long>()) {
      if (!tensor->data<long>()) {
        continue;
      }
      const long* data_orig = tensor->data<long>();
      data_temp.resize(tensor->numel());
      for (int j = 0; j < tensor->numel(); ++j) {
        data_temp[j] = data_orig[j];
      }
      data = data_temp.data();
    } else {
      if (!warning_printed_) {
        LOG(INFO) << "Tensor " << out_name << " has unsupported type "
                  << tensor->meta().name() << " with size " << tensor->numel();
        warning_printed_ = true;
      }
      continue;
    }

    info_->histograms[i].Add(data, tensor->numel());
    info_->total_histograms[i].Add(data, tensor->numel());
  }
  return;
}

OutputColumnMaxHistogramObserver::OutputColumnMaxHistogramObserver(
    OperatorBase* op,
    const std::string& col_max_blob_name,
    int nbins,
    // NOLINTNEXTLINE(modernize-pass-by-value)
    std::shared_ptr<HistogramObserver::Info> info)
    : ObserverBase<OperatorBase>(op),
      col_max_blob_name_(col_max_blob_name),
      nbins_(nbins),
      info_(info) {
  const auto& output_names = op->debug_def().output();
  auto it =
      std::find(output_names.begin(), output_names.end(), col_max_blob_name);
  CAFFE_ENFORCE(
      it != output_names.end(), "Cannot find blob in operator output.");
  col_max_blob_idx_ = std::distance(output_names.begin(), it);
};

void OutputColumnMaxHistogramObserver::Stop() {
  if (!subject_->OutputIsTensorType(col_max_blob_idx_, CPU)) {
    return;
  }
  Tensor* tensor = subject_->template Output<Tensor>(col_max_blob_idx_, CPU);
  if (!tensor || tensor->numel() == 0 || tensor->numel() == -1) {
    return;
  }

  float* data = GetFloatTensorData(tensor);
  if (data == nullptr && !warning_printed_) {
    LOG(INFO) << "Tensor " << col_max_blob_name_
              << " has mismatching type, or unsupported type "
              << tensor->meta().name() << " with size " << tensor->numel();
    warning_printed_ = true;
    return;
  }

  // determine number of columns
  CAFFE_ENFORCE(
      tensor->dim() == 2,
      "Tensor " + col_max_blob_name_ +
          " is not two-dimensional. Tensor.dim() = " +
          caffe2::to_string(tensor->dim()));
  int num_columns = tensor->size_from_dim(1);
  if (num_columns_ == -1) {
    num_columns_ = num_columns;
  }
  CAFFE_ENFORCE(
      num_columns_ == num_columns, "Observed inconsistent number of columns.");
  int num_rows = tensor->size_to_dim(1);
  for (int col = 0; col < num_columns; col++) {
    // find col max of the ith column
    auto col_max = std::abs(data[col]);
    for (int r = 0; r < num_rows; r++) {
      int idx = r * num_columns + col;
      col_max = max(col_max, std::abs(data[idx]));
    }
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    if (info_->histograms.size() <= col) {
      info_->histograms.emplace_back(nbins_);
      info_->total_histograms.emplace_back(nbins_);
      info_->min_max_info.tensor_infos.emplace_back(col_max_blob_name_);
    }
    info_->histograms[col].Add(col_max);
    info_->total_histograms[col].Add(col_max);
  }
}

HistogramNetObserver::HistogramNetObserver(
    NetBase* subject,
    // NOLINTNEXTLINE(modernize-pass-by-value)
    const string& out_file_name,
    int nbins,
    int dump_freq,
    bool mul_nets,
    string op_filter,
    // NOLINTNEXTLINE(modernize-pass-by-value)
    string delimiter)
    : NetObserver(subject),
      dump_freq_(dump_freq),
      cnt_(0),
      mul_nets_(mul_nets),
      op_filter_(op_filter),
      delimiter_(delimiter),
      out_file_name_(out_file_name) {
  net_name_ = subject->Name();
  if (op_filter != "") {
    bool has_op = false;
    for (auto* op : subject->GetOperators()) {
      if (op->debug_def().type() == op_filter) {
        has_op = true;
        break;
      }
    }
    if (!has_op) {
      LOG(INFO) << "Net " << net_name_ << " doesn't include operator "
                << op_filter;
      return;
    }
  }

  hist_infos_.resize(subject->GetOperators().size());

  int i = 0;
  for (auto* op : subject->GetOperators()) {
    shared_ptr<HistogramObserver::Info> info(new HistogramObserver::Info);
    info->min_max_info.type = op->debug_def().type();

    for (int j = 0; j < op->OutputSize(); ++j) {
      info->histograms.emplace_back(nbins);
      info->total_histograms.emplace_back(nbins);
      info->min_max_info.tensor_infos.emplace_back(op->debug_def().output(j));
    }

    HistogramObserver* observer = new HistogramObserver(op, info);
    op->AttachObserver(unique_ptr<HistogramObserver>(observer));
    hist_infos_[i] = info;
    ++i;
  }
}

void HistogramNetObserver::DumpAndReset_(
    const string& out_file_name,
    bool print_total_min_max) {
  if (hist_infos_.size() == 0) {
    return;
  }
  stringstream file_name;
  file_name << out_file_name;
  LOG(INFO) << "Dumping histograms of net " << net_name_ << " in " << this;
  if (mul_nets_) {
    file_name << ".";
    file_name << this;
  }
  ofstream f(file_name.str());
  if (!f) {
    LOG(WARNING) << this << ": can't open " << file_name.str();
  }

  // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
  for (int op_index = 0; op_index < hist_infos_.size(); ++op_index) {
    HistogramObserver::Info* info = hist_infos_[op_index].get();
    if (!info) {
      continue;
    }

    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    for (int i = 0; i < info->histograms.size(); ++i) {
      const Histogram* hist =
          (print_total_min_max ? info->total_histograms : info->histograms)[i]
              .Finalize();
      if (hist->Min() >= hist->Max()) {
        LOG(WARNING) << "Histogram of "
                     << info->min_max_info.tensor_infos[i].name
                     << " has an empty range: min " << hist->Min()
                     << " and max " << hist->Max();
      }
      if (hist->GetHistogram()->empty()) {
        LOG(WARNING) << "Histogram of "
                     << info->min_max_info.tensor_infos[i].name << " is empty";
      }

      ostringstream ost;
      ost << op_index << delimiter_ << info->min_max_info.type << delimiter_
          << i << delimiter_ << info->min_max_info.tensor_infos[i].name
          << delimiter_ << hist->Min() << delimiter_ << hist->Max()
          << delimiter_ << hist->GetHistogram()->size();

      for (uint64_t c : *hist->GetHistogram()) {
        ost << delimiter_ << c;
      }

      if (print_total_min_max) {
        LOG(INFO) << this << delimiter_ << ost.str();
      }

      f << ost.str() << endl;

      if (!print_total_min_max) {
        info->histograms[i] = DynamicHistogram(hist->GetHistogram()->size());
      }
    }
  }
  f.flush();
  f.close();
}

// NOLINTNEXTLINE(bugprone-exception-escape)
HistogramNetObserver::~HistogramNetObserver() {
  DumpAndReset_(out_file_name_, false);
}

void HistogramNetObserver::Stop() {
  ++cnt_;
  if (dump_freq_ == -1 || (cnt_ % dump_freq_) != 0) {
    return;
  }

  ostringstream ost;
  size_t last_dot = out_file_name_.rfind('.');
  size_t last_slash = out_file_name_.rfind('/');
  if (last_dot != string::npos &&
      (last_slash == string::npos || last_slash < last_dot)) {
    ost << out_file_name_.substr(0, last_dot) << "_" << cnt_ / dump_freq_
        << out_file_name_.substr(last_dot);
  } else {
    ost << out_file_name_ << "_" << cnt_ / dump_freq_;
  }

  DumpAndReset_(ost.str());
  return;
}

static bool HasDNNLowPEngine_(const OperatorDef& op_def) {
  const string ENGINE_PREFIX = "DNNLOWP";
  return strncmp(
             op_def.engine().c_str(),
             ENGINE_PREFIX.c_str(),
             ENGINE_PREFIX.size()) == 0;
}

static bool HasDNNLowPEngine_(const OperatorBase& op) {
  return HasDNNLowPEngine_(op.debug_def());
}

OutputColumnMaxHistogramNetObserver::OutputColumnMaxHistogramNetObserver(
    NetBase* subject,
    // NOLINTNEXTLINE(modernize-pass-by-value)
    const std::string& out_file_name,
    const std::vector<std::string>& observe_column_max_for_blobs,
    int nbins,
    int dump_freq,
    bool mul_nets,
    // NOLINTNEXTLINE(modernize-pass-by-value)
    string delimiter)
    : NetObserver(subject),
      dump_freq_(dump_freq),
      cnt_(0),
      mul_nets_(mul_nets),
      out_file_name_(out_file_name),
      delimiter_(delimiter) {
  if (observe_column_max_for_blobs.size() == 0) {
    return;
  }
  col_max_blob_names_.insert(
      observe_column_max_for_blobs.begin(), observe_column_max_for_blobs.end());
  int op_idx = 0;
  for (auto* op : subject->GetOperators()) {
    const auto& op_output_names = op->debug_def().output();
    int output_idx = 0;
    std::unordered_map<int, std::shared_ptr<HistogramObserver::Info>>
        output_col_hists_map;
    for (const auto& output_blob : op_output_names) {
      if (col_max_blob_names_.find(output_blob) == col_max_blob_names_.end()) {
        ++output_idx;
        continue;
      }
      /// create col max hist observer for blob
      auto info = std::make_shared<HistogramObserver::Info>();
      info->min_max_info.type = op->debug_def().type();
      // number of histograms in info will be determined at runtime by the
      // number of columns in the tensor.
      OutputColumnMaxHistogramObserver* observer =
          new OutputColumnMaxHistogramObserver(op, output_blob, nbins, info);
      op->AttachObserver(
          unique_ptr<OutputColumnMaxHistogramObserver>(observer));
      output_col_hists_map[output_idx] = info;
      ++output_idx;
    }
    if (output_col_hists_map.size() > 0) {
      hist_infos_[op_idx] = output_col_hists_map;
    }
    ++op_idx;
  }
}

void OutputColumnMaxHistogramNetObserver::DumpAndReset_(
    const std::string& out_file_name,
    bool print_total_min_max) {
  stringstream file_name;
  file_name << out_file_name;
  if (mul_nets_) {
    file_name << ".";
    file_name << this;
  }
  ofstream f(file_name.str());
  if (!f) {
    LOG(WARNING) << this << ": can't open " << file_name.str();
  }
  for (const auto& it : hist_infos_) {
    auto output_idx_hists_map = it.second;
    for (const auto& output_idx_hist : output_idx_hists_map) {
      int output_idx = output_idx_hist.first;
      HistogramObserver::Info* info = output_idx_hist.second.get();
      if (!info) {
        continue;
      }
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (int i = 0; i < info->histograms.size(); ++i) {
        const Histogram* hist =
            (print_total_min_max ? info->total_histograms : info->histograms)[i]
                .Finalize();
        if (hist->Min() >= hist->Max()) {
          LOG(WARNING) << "Histogram of "
                       << info->min_max_info.tensor_infos[i].name
                       << " has an empty range: min " << hist->Min()
                       << " and max " << hist->Max();
        }
        if (hist->GetHistogram()->empty()) {
          LOG(WARNING) << "Histogram of "
                       << info->min_max_info.tensor_infos[i].name
                       << " is empty";
        }
        ostringstream ost;
        // op_idx, output_idx, blob_name, col, min, max, nbins
        ost << it.first << delimiter_ << output_idx << delimiter_
            << info->min_max_info.tensor_infos[i].name << delimiter_ << i
            << delimiter_ << hist->Min() << delimiter_ << hist->Max()
            << delimiter_ << hist->GetHistogram()->size();

        // bins
        for (uint64_t c : *hist->GetHistogram()) {
          ost << delimiter_ << c;
        }
        if (print_total_min_max) {
          LOG(INFO) << this << delimiter_ << ost.str();
        }
        f << ost.str() << endl;
        if (!print_total_min_max) {
          info->histograms[i] = DynamicHistogram(hist->GetHistogram()->size());
        }
      }
    }
  }
  f.close();
}

void OutputColumnMaxHistogramNetObserver::Stop() {
  ++cnt_;
  if (dump_freq_ == -1 || (cnt_ % dump_freq_) != 0) {
    return;
  }
  ostringstream ost;
  size_t last_dot = out_file_name_.rfind('.');
  size_t last_slash = out_file_name_.rfind('/');
  if (last_dot != string::npos &&
      (last_slash == string::npos || last_slash < last_dot)) {
    ost << out_file_name_.substr(0, last_dot) << "_" << cnt_ / dump_freq_
        << out_file_name_.substr(last_dot);
  } else {
    ost << out_file_name_ << "_" << cnt_ / dump_freq_;
  }
  DumpAndReset_(ost.str());
  return;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
OutputColumnMaxHistogramNetObserver::~OutputColumnMaxHistogramNetObserver() {
  DumpAndReset_(out_file_name_, true);
}

RegisterQuantizationParamsNetObserver::RegisterQuantizationParamsNetObserver(
    NetBase* subject,
    const string& min_max_file_name,
    bool is_weight,
    const string& qparams_output_file_name)
    : NetObserver(subject) {
  ifstream f(min_max_file_name);

  // check the format by looking at the first line
  string first_line, word;
  getline(f, first_line);
  f.seekg(0, f.beg);
  istringstream ist(first_line);
  int nwords_first_line = 0;
  while (ist >> word) {
    ++nwords_first_line;
  }

  bool new_format = nwords_first_line == 6;
  if (!new_format && nwords_first_line != 5) {
    LOG(WARNING) << "min_max file " << min_max_file_name
                 << " has an invalid format";
  }

  // Optionally dump quantization params to file
  ofstream fout;
  if (!qparams_output_file_name.empty()) {
    fout.open(qparams_output_file_name);
    if (!fout) {
      LOG(WARNING) << this << ": can't open " << qparams_output_file_name;
    }
  }

  // parse the input file
  int op_index = 0;
  for (auto* op : subject->GetOperators()) {
    for (int i = 0; i < op->OutputSize(); ++i) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int op_index2, i2;
      string op_type, tensor_name;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      float min, max;

      if (new_format) {
        f >> op_index2 >> op_type >> i2 >> tensor_name >> min >> max;
      } else {
        f >> op_index2 >> i2 >> tensor_name >> min >> max;
      }
      assert(op_index2 == op_index);
      assert(i2 == i);
      assert(tensor_name == op->debug_def().output(i));

      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      TensorQuantizationParams qparams;
      if (max > min) {
        unique_ptr<QuantizationFactory> qfactory(GetQuantizationFactoryOf(op));
        qparams = qfactory->ChooseQuantizationParams(min, max, is_weight);
      } else {
        qparams.scale = 0.1f;
        qparams.zero_point = -min / qparams.scale;
        qparams.precision = 8;
      }

      if (HasDNNLowPEngine_(*op)) {
        SetStaticQuantizationParams(op, i, qparams);
      }

      if (fout.is_open()) {
        fout << op_index << " " << op_type << " " << i << " " << tensor_name
             << " " << qparams.Min() << " " << qparams.Max() << " "
             << qparams.scale << " " << qparams.zero_point << " "
             << qparams.precision << endl;
      }
    }
    ++op_index;
  }

  if (fout.is_open()) {
    fout.close();
  }
}

RegisterQuantizationParamsWithHistogramNetObserver::
    RegisterQuantizationParamsWithHistogramNetObserver(
        NetBase* subject,
        const string& histogram_file_name,
        bool is_weight,
        const string& qparams_output_file_name)
    : NetObserver(subject) {
  ifstream f(histogram_file_name);

  // check the format by looking at the first line
  string first_line, word;
  getline(f, first_line);
  f.seekg(0, f.beg);
  istringstream ist(first_line);
  int nwords_first_line = 0;
  while (ist >> word) {
    ++nwords_first_line;
  }

  ist.str(first_line);
  ist.clear();

  bool new_format = true;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int op_index, i, nbins;
  string op_type, tensor_name;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  float min, max;
  ist >> op_index >> op_type >> i >> tensor_name >> min >> max >> nbins;
  if (nwords_first_line != nbins + 7) {
    ist.str(first_line);
    ist.clear();
    ist >> op_index >> i >> tensor_name >> min >> max >> nbins;
    if (nwords_first_line == nbins + 6) {
      new_format = false;
    } else {
      LOG(WARNING) << "histogram file " << histogram_file_name
                   << " has an invalid format";
      return;
    }
  }

  // Optionally dump quantization params to file
  ofstream fout;
  if (!qparams_output_file_name.empty()) {
    fout.open(qparams_output_file_name);
    if (!fout) {
      LOG(WARNING) << this << ": can't open " << qparams_output_file_name;
    }
  }

  // parse the input file
  op_index = 0;
  for (auto* op : subject->GetOperators()) {
    for (i = 0; i < op->OutputSize(); ++i) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int op_index2, i2;

      if (new_format) {
        f >> op_index2 >> op_type >> i2 >> tensor_name >> min >> max >> nbins;
      } else {
        f >> op_index2 >> i2 >> tensor_name >> min >> max >> nbins;
      }
      LOG_IF(WARNING, op_index2 != op_index)
          << "op index " << op_index2 << " doesn't match with " << op_index;
      LOG_IF(WARNING, tensor_name != op->debug_def().output(i))
          << tensor_name << " in histogram file line " << op_index
          << " doesn't match with operation def " << op->debug_def().output(i);
      LOG_IF(WARNING, i2 != i)
          << "output tensor index " << i2 << " doesn't match with " << i;
      if (new_format) {
        LOG_IF(WARNING, op_type != op->debug_def().type())
            << "operator type " << op_type << " in histogram file line "
            << op_index << " doesn't match with operation def "
            << op->debug_def().type();
      }

      vector<uint64_t> bins;
      for (int j = 0; j < nbins; ++j) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        uint64_t cnt;
        f >> cnt;
        bins.push_back(cnt);
      }

      Histogram hist = Histogram(min, max, bins);

      LOG(INFO) << "Choosing qparams for " << tensor_name;
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      TensorQuantizationParams qparams;
      if (max > min) {
        unique_ptr<QuantizationFactory> qfactory(GetQuantizationFactoryOf(op));
        qparams = qfactory->ChooseQuantizationParams(hist, is_weight);
      } else {
        qparams.scale = 0.1f;
        qparams.precision = 8;
        qparams.zero_point =
            (isinf(min / qparams.scale) || isnan(min / qparams.scale))
            ? 0
            : std::max(
                  0,
                  std::min(
                      int((-min) / qparams.scale),
                      (1 << qparams.precision) - 1));
      }

      if (HasDNNLowPEngine_(*op)) {
        SetStaticQuantizationParams(op, i, qparams);
      }

      if (fout.is_open()) {
        fout << op_index << " " << op_type << " " << i << " " << tensor_name
             << " " << qparams.Min() << " " << qparams.Max() << " "
             << qparams.scale << " " << qparams.zero_point << " "
             << qparams.precision << endl;
      }
    }
    ++op_index;
  }

  if (fout.is_open()) {
    fout.close();
  }
}

} // namespace caffe2
