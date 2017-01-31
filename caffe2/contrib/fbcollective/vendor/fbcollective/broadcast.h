#pragma once

#include "fbcollective/algorithm.h"
#include "fbcollective/common/logging.h"

namespace fbcollective {

template <typename T>
class Broadcast : public Algorithm {
 public:
  Broadcast(const std::shared_ptr<Context>& context, int rootRank)
      : Algorithm(context), rootRank_(rootRank) {
    FBC_ENFORCE_GE(rootRank_, 0);
    FBC_ENFORCE_LT(rootRank_, contextSize_);
  }

  virtual ~Broadcast(){};

 protected:
  const int rootRank_;
};

} // namespace fbcollective
