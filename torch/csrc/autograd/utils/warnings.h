#pragma once
#include <c10/util/Exception.h>

#include <vector>
#include <mutex>


namespace torch { namespace autograd { namespace utils {

// Warning handler for multi-threaded contexts. Gather warnings from
// all threads into a single queue, then process together at the end
// in the main thread.
class DelayWarningHandler: public at::WarningHandler {
public:
  ~DelayWarningHandler() override = default;
  void replay_warnings();

private:

  void process(const at::SourceLocation &source_location,
               const std::string &msg,
               bool verbatim) override;

  struct Warning {
    c10::SourceLocation source_location;
    std::string msg;
    bool verbatim;
  };

  std::vector<Warning> warnings_;
  std::mutex mutex_;
};

}}}  // namespace torch::autograd::utils
