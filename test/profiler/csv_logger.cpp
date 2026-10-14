#include <torch/extension.h>

#ifdef USE_KINETO

#include <fstream>
#include <string>

#include <libkineto.h>
#include <output_base.h>
#include <torch/csrc/profiler/standalone/custom_logger_registry.h>

namespace torch::profiler::impl {

class CsvLogger : public libkineto::ActivityLogger {
 public:
  explicit CsvLogger(const std::string& filename) : out_(filename) {
    if (out_.is_open()) {
      out_ << "name,start_us,duration_us,device_id,resource_id,correlation_id"
           << std::endl;
    }
  }

  ~CsvLogger() override {
    if (out_.is_open()) {
      out_.close();
    }
  }

  void handleDeviceInfo(const libkineto::DeviceInfo&, int64_t) override {}
  void handleResourceInfo(const libkineto::ResourceInfo&, int64_t) override {}
  void handleOverheadInfo(const OverheadInfo&, int64_t) override {}
  void handleTraceSpan(const libkineto::TraceSpan&) override {}

  void handleActivity(const libkineto::ITraceActivity& activity) override {
    if (!out_.is_open()) {
      return;
    }
    const std::string& name = activity.name();
    if (name.find(',') != std::string::npos) {
      out_ << '"' << name << '"';
    } else {
      out_ << name;
    }
    out_ << ',' << libkineto::ITraceActivity::nsToUs(activity.timestamp())
         << ',' << libkineto::ITraceActivity::nsToUs(activity.duration())
         << ',' << activity.deviceId()
         << ',' << activity.resourceId()
         << ',' << activity.correlationId()
         << std::endl;
  }

  void handleGenericActivity(
      const libkineto::GenericTraceActivity& activity) override {
    handleActivity(activity);
  }

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>&,
      const std::string&) override {}

  void finalizeMemoryTrace(
      const std::string&,
      const libkineto::Config&) override {}

  void finalizeTrace(
      const libkineto::Config&,
      std::unique_ptr<libkineto::ActivityBuffers>,
      int64_t,
      std::unordered_map<std::string, std::vector<std::string>>&) override {}

 private:
  std::ofstream out_;
};

REGISTER_CUSTOM_LOGGER("csv", CsvLogger);

} // namespace torch::profiler::impl

#endif // USE_KINETO

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
