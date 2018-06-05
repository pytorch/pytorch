#include <TH/THStorage.h>
#include <cstdint>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../../process_group/General.hpp"
#include "../common/Functions.hpp"
#include "../common/RPC.hpp"
#include "../master/Master.hpp"
#include "Worker.hpp"

namespace thd {
namespace worker {

namespace detail {

void sendValueToMaster(int64_t value) {
  IntScalar scalar(value);
  dataChannel->send(scalar, 0);
}

void sendValueToMaster(double value) {
  FloatScalar scalar(value);
  dataChannel->send(scalar, 0);
}

at::Tensor& unpackRetrieveTensor(rpc::RPCMessage& message) {
  return workerTensors.at(unpackTensor(message));
}

at::Storage* unpackRetrieveStorage(rpc::RPCMessage& message) {
  return workerStorages.at(unpackStorage(message)).get();
}

at::Generator* unpackRetrieveGenerator(rpc::RPCMessage& message) {
  return workerGenerators.at(unpackGenerator(message)).get();
}

static void finalize(rpc::RPCMessage& raw_message) {
  if (raw_message.remaining() > 0)
    throw std::invalid_argument("message is too long");
}

#include "dispatch/Communication.cpp"
#include "dispatch/Generator.cpp"
#include "dispatch/Storage.cpp"
#include "dispatch/Tensor.cpp"
#include "dispatch/TensorCopy.cpp"
#include "dispatch/TensorMath.cpp"
#include "dispatch/TensorRandom.cpp"
#include "dispatch/TensorLapack.cpp"

using dispatch_fn = void (*)(rpc::RPCMessage&);
using Functions = thd::Functions;

void exitWorker(rpc::RPCMessage& msg) {
  finalize(msg);
  ::exit(0);
}


static const std::unordered_map<rpc::function_id_type, dispatch_fn> functions {
    {Functions::generatorNew, generatorNew},
    {Functions::generatorFree, generatorFree},
    {Functions::generatorCopy, generatorCopy},
    {Functions::generatorSeed, generatorSeed},
    {Functions::generatorManualSeed, generatorManualSeed},

    {Functions::tensorCopyFromMaster, tensorCopyFromMaster},
    {Functions::tensorCopyFromWorker, tensorCopyFromWorker},

    {Functions::tensorNew, tensorNew},
    {Functions::tensorNewWithSize, tensorNewWithSize},
    {Functions::tensorNewWithStorage, tensorNewWithStorage},
    {Functions::tensorResize, tensorResize},
    {Functions::tensorResizeAs, tensorResizeAs},
    {Functions::tensorResize1d, tensorResize1d},
    {Functions::tensorResize2d, tensorResize2d},
    {Functions::tensorResize3d, tensorResize2d},
    {Functions::tensorResize4d, tensorResize2d},
    {Functions::tensorResize5d, tensorResize2d},
    {Functions::tensorSetStorage, tensorSetStorage},
    {Functions::tensorSetStorage1d, tensorSetStorage1d},
    {Functions::tensorSetStorage2d, tensorSetStorage2d},
    {Functions::tensorSetStorage3d, tensorSetStorage3d},
    {Functions::tensorSetStorage4d, tensorSetStorage4d},
    {Functions::tensorNarrow, tensorNarrow},
    {Functions::tensorSelect, tensorSelect},
    {Functions::tensorTranspose, tensorTranspose},
    {Functions::tensorUnfold, tensorUnfold},
    {Functions::tensorSqueeze, tensorSqueeze},
    {Functions::tensorSqueeze, tensorSqueeze1d},

    {Functions::tensorFree, tensorFree},
    {Functions::tensorAdd, tensorAdd},

    {Functions::tensorGather, tensorGather},
    {Functions::tensorScatter, tensorScatter},
    {Functions::tensorScatterFill, tensorScatterFill},
    {Functions::tensorDot, tensorDot},
    {Functions::tensorMinall, tensorMinall},
    {Functions::tensorMaxall, tensorMaxall},
    {Functions::tensorMedianall, tensorMedianall},
    {Functions::tensorSumall, tensorSumall},
    {Functions::tensorProdall, tensorProdall},
    {Functions::tensorNeg, tensorNeg},
    {Functions::tensorCinv, tensorCinv},
    {Functions::tensorAdd, tensorAdd},
    {Functions::tensorSub, tensorSub},
    {Functions::tensorMul, tensorMul},
    {Functions::tensorDiv, tensorDiv},
    {Functions::tensorFmod, tensorFmod},
    {Functions::tensorRemainder, tensorRemainder},
    {Functions::tensorClamp, tensorClamp},
    {Functions::tensorCadd, tensorCadd},
    {Functions::tensorCsub, tensorCsub},
    {Functions::tensorCmul, tensorCmul},
    {Functions::tensorCpow, tensorCpow},
    {Functions::tensorCdiv, tensorCdiv},
    {Functions::tensorCfmod, tensorCfmod},
    {Functions::tensorCremainder, tensorCremainder},
    {Functions::tensorAddcmul, tensorAddcmul},
    {Functions::tensorAddcdiv, tensorAddcdiv},
    {Functions::tensorAddmv, tensorAddmv},
    {Functions::tensorAddmm, tensorAddmm},
    {Functions::tensorAddr, tensorAddr},
    {Functions::tensorAddbmm, tensorAddbmm},
    {Functions::tensorBaddbmm, tensorBaddbmm},
    /* {Functions::tensorMatch, tensorMatch}, */
    {Functions::tensorMax, tensorMax},
    {Functions::tensorMin, tensorMin},
    {Functions::tensorKthvalue, tensorKthvalue},
    {Functions::tensorMode, tensorMode},
    {Functions::tensorMedian, tensorMedian},
    {Functions::tensorSum, tensorSum},
    {Functions::tensorProd, tensorProd},
    {Functions::tensorCumsum, tensorCumsum},
    {Functions::tensorCumprod, tensorCumprod},
    {Functions::tensorSign, tensorSign},
    {Functions::tensorTrace, tensorTrace},
    {Functions::tensorCross, tensorCross},
    {Functions::tensorCmax, tensorCmax},
    {Functions::tensorCmin, tensorCmin},
    /* {Functions::tensorCmaxValue, tensorCmaxValue}, */
    /* {Functions::tensorCminValue, tensorCminValue}, */

    {Functions::tensorFill, tensorFill},
    {Functions::tensorMaskedFill, tensorMaskedFill},
    {Functions::tensorMaskedCopy, tensorMaskedCopy},
    {Functions::tensorMaskedSelect, tensorMaskedSelect},
    {Functions::tensorNonzero, tensorNonzero},
    {Functions::tensorIndexSelect, tensorIndexSelect},
    {Functions::tensorIndexCopy, tensorIndexCopy},
    {Functions::tensorIndexAdd, tensorIndexAdd},
    {Functions::tensorIndexFill, tensorIndexFill},
    {Functions::tensorDiag, tensorDiag},
    {Functions::tensorEye, tensorEye},
    {Functions::tensorRange, tensorRange},
    {Functions::tensorRandperm, tensorRandperm},
    {Functions::tensorSort, tensorSort},
    {Functions::tensorTopk, tensorTopk},
    {Functions::tensorTril, tensorTril},
    {Functions::tensorTriu, tensorTriu},
    {Functions::tensorEqual, tensorEqual},
    {Functions::tensorLtValue, tensorLtValue},
    {Functions::tensorLeValue, tensorLeValue},
    {Functions::tensorGtValue, tensorGtValue},
    {Functions::tensorGeValue, tensorGeValue},
    {Functions::tensorNeValue, tensorNeValue},
    {Functions::tensorEqValue, tensorEqValue},
    {Functions::tensorLtValueT, tensorLtValueT},
    {Functions::tensorLeValueT, tensorLeValueT},
    {Functions::tensorGtValueT, tensorGtValueT},
    {Functions::tensorGeValueT, tensorGeValueT},
    {Functions::tensorNeValueT, tensorNeValueT},
    {Functions::tensorEqValueT, tensorEqValueT},
    {Functions::tensorLtTensor, tensorLtTensor},
    {Functions::tensorLeTensor, tensorLeTensor},
    {Functions::tensorGtTensor, tensorGtTensor},
    {Functions::tensorGeTensor, tensorGeTensor},
    {Functions::tensorNeTensor, tensorNeTensor},
    {Functions::tensorEqTensor, tensorEqTensor},
    {Functions::tensorLtTensorT, tensorLtTensorT},
    {Functions::tensorLeTensorT, tensorLeTensorT},
    {Functions::tensorGtTensorT, tensorGtTensorT},
    {Functions::tensorGeTensorT, tensorGeTensorT},
    {Functions::tensorNeTensorT, tensorNeTensorT},
    {Functions::tensorEqTensorT, tensorEqTensorT},
    {Functions::tensorAbs, tensorAbs},
    {Functions::tensorSigmoid, tensorSigmoid},
    {Functions::tensorLog, tensorLog},
    {Functions::tensorLog10, tensorLog10},
    {Functions::tensorLog1p, tensorLog1p},
    {Functions::tensorLog2, tensorLog2},
    {Functions::tensorExp, tensorExp},
    {Functions::tensorExpm1, tensorExpm1},
    {Functions::tensorCos, tensorCos},
    {Functions::tensorAcos, tensorAcos},
    {Functions::tensorCosh, tensorCosh},
    {Functions::tensorSin, tensorSin},
    {Functions::tensorAsin, tensorAsin},
    {Functions::tensorSinh, tensorSinh},
    {Functions::tensorTan, tensorTan},
    {Functions::tensorAtan, tensorAtan},
    {Functions::tensorAtan2, tensorAtan2},
    {Functions::tensorTanh, tensorTanh},
    {Functions::tensorPow, tensorPow},
    {Functions::tensorTpow, tensorTpow},
    {Functions::tensorSqrt, tensorSqrt},
    {Functions::tensorRsqrt, tensorRsqrt},
    {Functions::tensorCeil, tensorCeil},
    {Functions::tensorFloor, tensorFloor},
    {Functions::tensorRound, tensorRound},
    {Functions::tensorTrunc, tensorTrunc},
    {Functions::tensorFrac, tensorFrac},
    {Functions::tensorLerp, tensorLerp},
    {Functions::tensorMean, tensorMean},
    {Functions::tensorStd, tensorStd},
    {Functions::tensorVar, tensorVar},
    {Functions::tensorNorm, tensorNorm},
    {Functions::tensorRenorm, tensorRenorm},
    {Functions::tensorDist, tensorDist},
    {Functions::tensorHistc, tensorHistc},
    /* {Functions::tensorBhistc, tensorBhistc}, */
    {Functions::tensorMeanall, tensorMeanall},
    {Functions::tensorVarall, tensorVarall},
    {Functions::tensorStdall, tensorStdall},
    {Functions::tensorNormall, tensorNormall},
    {Functions::tensorLinspace, tensorLinspace},
    {Functions::tensorLogspace, tensorLogspace},
    {Functions::tensorRand, tensorRand},
    {Functions::tensorRandn, tensorRandn},
    {Functions::tensorLogicalAndAll, tensorLogicalAndAll},
    {Functions::tensorLogicalAnd, tensorLogicalAnd},
    {Functions::tensorLogicalAnyAll, tensorLogicalAnyAll},
    {Functions::tensorLogicalAny, tensorLogicalAny},
    {Functions::tensorRandom, tensorRandom},
    {Functions::tensorGeometric, tensorGeometric},
    {Functions::tensorBernoulli, tensorBernoulli},
    {Functions::tensorBernoulli_FloatTensor, tensorBernoulli_FloatTensor},
    {Functions::tensorBernoulli_DoubleTensor, tensorBernoulli_DoubleTensor},
    {Functions::tensorUniform, tensorUniform},
    {Functions::tensorNormal, tensorNormal},
    {Functions::tensorExponential, tensorExponential},
    {Functions::tensorCauchy, tensorCauchy},
    {Functions::tensorLogNormal, tensorLogNormal},
    {Functions::tensorMultinomial, tensorMultinomial},

    {Functions::tensorGesv, tensorGesv},
    {Functions::tensorTrtrs, tensorTrtrs},
    {Functions::tensorGels, tensorGels},
    {Functions::tensorSyev, tensorSyev},
    {Functions::tensorGeev, tensorGeev},
    {Functions::tensorGesvd2, tensorGesvd2},
    {Functions::tensorGetri, tensorGetri},
    {Functions::tensorPotrf, tensorPotrf},
    {Functions::tensorPotrs, tensorPotrs},
    {Functions::tensorPotri, tensorPotri},
    {Functions::tensorQr, tensorQr},
    {Functions::tensorGeqrf, tensorGeqrf},
    {Functions::tensorOrgqr, tensorOrgqr},
    {Functions::tensorOrmqr, tensorOrmqr},
    {Functions::tensorPstrf, tensorPstrf},

    {Functions::storageNew, storageNew},
    {Functions::storageNewWithSize, storageNewWithSize},
    {Functions::storageNewWithSize1, storageNewWithSize1},
    {Functions::storageNewWithSize2, storageNewWithSize2},
    {Functions::storageNewWithSize3, storageNewWithSize3},
    {Functions::storageNewWithSize4, storageNewWithSize4},
    {Functions::storageFree, storageFree},
    {Functions::storageResize, storageResize},
    {Functions::storageFill, storageFill},

    {Functions::sendTensor, sendTensor},
    {Functions::sendStorage, sendStorage},

    {Functions::exit, exitWorker}
};

} // namespace detail

/* On fail throws exceptions which should be caught in worker's loop and reported
 * to master.
 */
void execute(std::unique_ptr<rpc::RPCMessage> raw_message_ptr) {
  auto &raw_message = *raw_message_ptr;
  rpc::function_id_type fid = rpc::unpackFunctionId(raw_message);
  auto iter = detail::functions.find(fid);
  if (iter != detail::functions.end()) {
    (*iter->second)(raw_message);
  } else {
    throw std::invalid_argument("invalid function id: " + std::to_string(fid));
  }
}

} // namespace worker
} // namespace thd
