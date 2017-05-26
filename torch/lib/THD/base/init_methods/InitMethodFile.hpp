#pragma once

#include "InitMethod.hpp"

namespace thd {

struct InitMethodFile : InitMethod {
  InitMethodFile(std::string file_path, rank_type world_size);
  virtual ~InitMethodFile();

  InitMethod::Config getConfig() override;

private:
  std::string _file_path;
  rank_type _world_size;
  int _file;
};

} // namespace thd
