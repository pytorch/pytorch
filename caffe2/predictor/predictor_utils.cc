#include "caffe2/predictor/predictor_utils.h"
#include "caffe2/predictor/predictor_config.h"

#include "caffe2/core/blob.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/predictor_consts.pb.h"
#include "caffe2/utils/proto_utils.h"

CAFFE2_DEFINE_bool(
    caffe2_predictor_claim_tensor_memory,
    true,
    "If false, then predictor will not claim tensor memory"
    "otherwise when tensor is shrinked to a size smaller than current size "
    "by FLAGS_caffe2_max_keep_on_shrink_memory, the memory will be claimed.");

namespace caffe2 {
namespace predictor_utils {

CAFFE2_API const NetDef& getNet(
    const MetaNetDef& def,
    const std::string& name) {
  for (const auto& n : def.nets()) {
    if (n.key() == name) {
      return n.value();
    }
  }
  CAFFE_THROW("Net not found: ", name);
}

std::unique_ptr<MetaNetDef> extractMetaNetDef(
    db::Cursor* cursor,
    const std::string& key) {
  CAFFE_ENFORCE(cursor);
  if (cursor->SupportsSeek()) {
    cursor->Seek(key);
  }
  for (; cursor->Valid(); cursor->Next()) {
    if (cursor->key() != key) {
      continue;
    }
    // We've found a match. Parse it out.
    BlobProto proto;
    CAFFE_ENFORCE(proto.ParseFromString(cursor->value()));
    Blob blob;
    blob.Deserialize(proto);
    CAFFE_ENFORCE(blob.template IsType<string>());
    auto def = caffe2::make_unique<MetaNetDef>();
    CAFFE_ENFORCE(def->ParseFromString(blob.template Get<string>()));
    return def;
  }
  CAFFE_THROW("Failed to find in db the key: ", key);
}

std::unique_ptr<MetaNetDef> runGlobalInitialization(
    std::unique_ptr<db::DBReader> db,
    Workspace* master) {
  CAFFE_ENFORCE(db.get());
  auto* cursor = db->cursor();

  auto metaNetDef = extractMetaNetDef(
      cursor, PredictorConsts::default_instance().meta_net_def());
  if (metaNetDef->has_modelinfo()) {
    CAFFE_ENFORCE(
        metaNetDef->modelinfo().predictortype() ==
            PredictorConsts::default_instance().single_predictor(),
        "Can only load single predictor");
  }
  VLOG(1) << "Extracted meta net def";

  const auto globalInitNet = getNet(
      *metaNetDef, PredictorConsts::default_instance().global_init_net_type());
  VLOG(1) << "Global init net: " << ProtoDebugString(globalInitNet);

  // Now, pass away ownership of the DB into the master workspace for
  // use by the globalInitNet.
  master->CreateBlob(PredictorConsts::default_instance().predictor_dbreader())
      ->Reset(db.release());

  // Now, with the DBReader set, we can run globalInitNet.
  CAFFE_ENFORCE(
      master->RunNetOnce(globalInitNet),
      "Failed running the globalInitNet: ",
      ProtoDebugString(globalInitNet));

  return metaNetDef;
}

} // namespace predictor_utils

void removeExternalBlobs(
    const std::vector<std::string>& input_blobs,
    const std::vector<std::string>& output_blobs,
    Workspace* ws) {
  for (const auto& blob : input_blobs) {
    ws->RemoveBlob(blob);
  }
  for (const auto& blob : output_blobs) {
    ws->RemoveBlob(blob);
  }
}

PredictorConfig makePredictorConfig(
    const string& db_type,
    const string& db_path) {
  // TODO: Remove this flags once Predictor accept PredictorConfig as
  // constructors. These comes are copied temporarly from the Predictor.
  if (FLAGS_caffe2_predictor_claim_tensor_memory) {
    if (FLAGS_caffe2_max_keep_on_shrink_memory == LLONG_MAX) {
      FLAGS_caffe2_max_keep_on_shrink_memory = 8 * 1024 * 1024;
    }
  }
  auto dbReader =
      make_unique<db::DBReader>(db::CreateDB(db_type, db_path, db::READ));
  auto ws = std::make_shared<Workspace>();
  auto net_def =
      predictor_utils::runGlobalInitialization(std::move(dbReader), ws.get());
  auto config = makePredictorConfig(*net_def, ws.get());
  config.ws = ws;
  const auto& init_net = predictor_utils::getNet(
      *net_def, PredictorConsts::default_instance().predict_init_net_type());
  CAFFE_ENFORCE(config.ws->RunNetOnce(init_net));
  config.ws->RemoveBlob(
      PredictorConsts::default_instance().predictor_dbreader());
  // Input and output blobs should never be allocated in the master workspace
  // since we'll end up with race-conditions due to these being shared among
  // predictor threads / TL workspaces. Safely handle against globalInitNet
  // creating them in the master.
  removeExternalBlobs(config.input_names, config.output_names, config.ws.get());
  return config;
}

} // namespace caffe2
