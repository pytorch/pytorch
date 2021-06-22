#pragma once

#include <c10/util/Optional.h>

#include <iostream>
#include <string>
#include <vector>

namespace torch_lazy_tensors {

struct SourceLocation {
  std::string file;
  std::string function;
  int line = -1;
};

c10::optional<SourceLocation> GetPythonFrameTop();

std::vector<SourceLocation> GetPythonFrames();

std::ostream& operator<<(std::ostream& stream,
                         const std::vector<SourceLocation>& frames);

}  // namespace torch_lazy_tensors
