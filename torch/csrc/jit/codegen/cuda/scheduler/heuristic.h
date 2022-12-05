#pragma once

#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <string>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class HeuristicParams : public PolymorphicBase {
 public:
  std::string tag = "";

  LaunchParams lparams;

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

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
