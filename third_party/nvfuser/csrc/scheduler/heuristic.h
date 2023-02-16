#pragma once

#include <executor_params.h>
#include <utils.h>

#include <string>

namespace nvfuser {

class HeuristicParams : public PolymorphicBase {
 public:
  std::string tag = "";

  LaunchParams lparams;
  CompileParams cparams;

  virtual std::string toString() const {
    return "Undefined Heuristic Params";
  }

  virtual size_t hash() const = 0;

  virtual ~HeuristicParams() = default;

  virtual bool sameAs(const std::shared_ptr<HeuristicParams>& other) const = 0;

  virtual std::shared_ptr<HeuristicParams> clone() const = 0;

  HeuristicParams() = default;
  HeuristicParams(const std::string& tag) : tag(tag) {}
};

} // namespace nvfuser
