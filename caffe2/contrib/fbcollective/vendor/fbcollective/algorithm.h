#pragma once

#include <memory>

#include "fbcollective/context.h"

namespace fbcollective {

class Algorithm {
 public:
  explicit Algorithm(const std::shared_ptr<Context>&);
  virtual ~Algorithm() = 0;

  virtual void Run() = 0;

 protected:
  std::shared_ptr<Context> context_;

  const int contextRank_;
  const int contextSize_;

  std::unique_ptr<transport::Pair>& getPair(int i);

  // Helpers for ring algorithms
  std::unique_ptr<transport::Pair>& getLeftPair();
  std::unique_ptr<transport::Pair>& getRightPair();
};

} // namespace fbcollective
