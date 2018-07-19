#pragma once

#include <caffe2/distributed/store_handler.h>

namespace caffe2 {

class FileStoreHandler : public StoreHandler {
 public:
  explicit FileStoreHandler(const std::string& path, const std::string& prefix);
  virtual ~FileStoreHandler();

  virtual void set(const std::string& name, const std::string& data) override;

  virtual std::string get(const std::string& name) override;

  virtual int64_t add(const std::string& name, int64_t value) override;

  virtual bool check(const std::vector<std::string>& names) override;

  virtual void wait(
      const std::vector<std::string>& names,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) override;

 protected:
  std::string basePath_;

  std::string realPath(const std::string& path);

  std::string tmpPath(const std::string& name);

  std::string objectPath(const std::string& name);
};

} // namespace caffe2
