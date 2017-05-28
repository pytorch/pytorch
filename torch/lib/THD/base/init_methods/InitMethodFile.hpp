#pragma once

#include "InitMethod.hpp"

namespace thd {

struct InitMethodFile : InitMethod {
  InitMethodFile(std::string file_path, rank_type world_size, std::string group_name);
  virtual ~InitMethodFile();

  InitMethod::Config getConfig() override;

private:
  std::string _file_path;
  std::string _group_name;
  rank_type _world_size;
  int _file;
};

} // namespace thd
