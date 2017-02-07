#pragma once

#include "fbcollective/algorithm.h"
#include "fbcollective/common/logging.h"

namespace fbcollective {

class Barrier : public Algorithm {
 public:
  explicit Barrier(const std::shared_ptr<Context>& context)
      : Algorithm(context) {}

  virtual ~Barrier(){};
};

} // namespace fbcollective
