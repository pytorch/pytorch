#include "caffe2/core/logging.h"
#include "caffe2/opt/custom/pointwise_elim.h"
#include "caffe2/opt/nql/graphmatcher.h"
#include "caffe2/opt/passes.h"
#include "nomnigraph/Representations/NeuralNet.h"
#include "nomnigraph/Support/Common.h"
#include "nomnigraph/Transformations/SubgraphMatcher.h"

namespace caffe2 {
namespace opt {

using namespace nom::repr;

void fuseCastBatchOneHot(NNModule* nn) {
  nom::nql::GraphMatcher gm;
  gm.initFromString(R"NQL(def nn {
      %cast = Cast(%input)
      %one_hot = BatchOneHot(%cast, %lengths, %values)
      %out = Cast(%one_hot)
  })NQL");
  CAFFE_ENFORCE(gm.getMatcher(), "Unable to parse NQL query.");

  for (const auto& match : gm.getMatches(nn->dataFlow)) {
    // This matches most of prod as of H2 2018
    auto first_cast = nn::getProducer(match["\%cast"]);
    auto second_cast = nn::getProducer(match["\%out"]);
    NOM_REQUIRE_OR_CONT(nn::get<Cast>(first_cast)->getTo() == 10);
    NOM_REQUIRE_OR_CONT(nn::get<Cast>(second_cast)->getTo() == 1);

    nn->replaceSubgraphWithOperator<CastedBatchOneHot>(
        match.subgraph,
        {match["\%input"], match["\%lengths"], match["\%values"]},
        {match["\%out"]});
  }
}

REGISTER_OPT_PASS_FROM_FUNC(FuseCastBatchOneHot, fuseCastBatchOneHot);

} // namespace opt
} // namespace caffe2
