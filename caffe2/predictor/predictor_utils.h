#pragma once

#include "caffe2/core/db.h"
#include "caffe2/core/workspace.h"
#include "caffe2/predictor/predictor_config.h"
#include "caffe2/proto/metanet.pb.h"

namespace caffe2 {
namespace predictor_utils {

TORCH_API const NetDef& getNet(const MetaNetDef& def, const std::string& name);
const ::google::protobuf::RepeatedPtrField<::std::string>& getBlobs(
    const MetaNetDef& def,
    const std::string& name);

TORCH_API std::unique_ptr<MetaNetDef> extractMetaNetDef(
    db::Cursor* cursor,
    const std::string& key);

// Extract the MetaNetDef from `db`, and run the global init net on the
// `master` workspace.
TORCH_API std::unique_ptr<MetaNetDef> runGlobalInitialization(
    std::unique_ptr<db::DBReader> db,
    Workspace* master);

} // namespace predictor_utils
} // namespace caffe2
