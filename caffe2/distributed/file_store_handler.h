/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
