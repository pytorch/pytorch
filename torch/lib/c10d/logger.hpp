#include <c10d/reducer.hpp>

namespace c10d {

class Logger {
 public:
 explicit Logger(std::shared_ptr<c10d::Reducer> reducer);
  // Set logging data that can be got during DistributedDataParallel
  // construction time.
  void set_construction_logging_data(
      const std::string& module_name,
      const std::vector<int>& device_ids,
      int output_device,
      bool broadcast_buffers);

  // An Interface for users to get DDPLoggingData and log them
  // in the applications.
  c10::DDPLoggingData get_ddp_logging_data();

  // Set environment variables.
  void set_env_variables();
  // Set parameters stats.
  void set_parameter_stats();
  // Set buckets related stats.
  void set_bucket_stats();

 private:
  // ddp_logging_data_ is used to hold all the ddp related logging
  // data fields.
  std::unique_ptr<c10::DDPLoggingData> ddp_logging_data_;
  std::shared_ptr<c10d::Reducer> reducer_;
};

} // namespace c10d
