#include "caffe2/opt/converter.h"
#include "caffe2/utils/cast.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {
namespace {

using namespace nom;
using namespace nom::repr;

class BatchMatMulConverter : public Converter {
  std::unique_ptr<repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::BatchMatMul>();
    auto argMap = getArgumentsFromOperator(op);

    auto c = dyn_cast<repr::BatchMatMul>(nnOp.get());
    if (argMap.count("trans_a")) {
      CAFFE_ENFORCE(argMap["trans_a"].has_i(), "Invalid axis argument");
      int trans_a = static_cast<int>(argMap["trans_a"].i());
      c->setTransA(!!trans_a);
    }
    if (argMap.count("trans_b")) {
      CAFFE_ENFORCE(argMap["trans_b"].has_i(), "Invalid add_axis argument");
      int trans_b = static_cast<int>(argMap["trans_b"].i());
      c->setTransB(!!trans_b);
    }
    if (argMap.count("broadcast")) {
      CAFFE_ENFORCE(argMap["broadcast"].has_i(), "Invalid add_axis argument");
      int broadcast = static_cast<int>(argMap["broadcast"].i());
      c->setBroadcast(!!broadcast);
    }
    return nnOp;
  }
  // Does not override default converter to OperatorDef

  ~BatchMatMulConverter() override {}
};
REGISTER_CONVERTER(BatchMatMul, BatchMatMulConverter);

TRIVIAL_CONVERTER(BatchGather);
REGISTER_CONVERTER(BatchGather, BatchGatherConverter);

class MulConverter : public Converter {
  std::unique_ptr<repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::Mul>();
    auto argMap = getArgumentsFromOperator(op);

    auto c = dyn_cast<repr::Mul>(nnOp.get());
    if (argMap.count("broadcast")) {
      CAFFE_ENFORCE(argMap["broadcast"].has_i(), "Invalid broadcast argument");
      int broadcast = static_cast<int>(argMap["broadcast"].i());
      c->setBroadcast(!!broadcast);
    }
    return nnOp;
  }
  // Does not override default converter to OperatorDef

  ~MulConverter() override {}
};
REGISTER_CONVERTER(Mul, MulConverter);

class AddConverter : public Converter {
  std::unique_ptr<repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::Add>();
    auto argMap = getArgumentsFromOperator(op);

    auto c = dyn_cast<repr::Add>(nnOp.get());
    if (argMap.count("broadcast")) {
      CAFFE_ENFORCE(argMap["broadcast"].has_i(), "Invalid broadcast argument");
      int broadcast = static_cast<int>(argMap["broadcast"].i());
      c->setBroadcast(!!broadcast);
    }
    return nnOp;
  }
  // Does not override default converter to OperatorDef

  ~AddConverter() override {}
};
REGISTER_CONVERTER(Add, AddConverter);

class CastConverter : public Converter {
  std::unique_ptr<repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::Cast>();
    auto argMap = getArgumentsFromOperator(op);

    auto c = dyn_cast<repr::Cast>(nnOp.get());
    ArgumentHelper helper(op);
    c->setTo(cast::GetCastDataType(helper, "to"));
    return nnOp;
  }
  // Does not override default converter to OperatorDef

  ~CastConverter() override {}
};
REGISTER_CONVERTER(Cast, CastConverter);

class ReplaceNaNConverter : public Converter {
  std::unique_ptr<repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::ReplaceNaN>();
    auto argMap = getArgumentsFromOperator(op);

    auto c = dyn_cast<repr::ReplaceNaN>(nnOp.get());
    if (argMap.count("value")) {
      CAFFE_ENFORCE(argMap["value"].has_f(), "Invalid 'value' argument");
      float value = static_cast<float>(argMap["value"].f());
      c->setValue(value);
    }
    return nnOp;
  }
  // Does not override default converter to OperatorDef

  ~ReplaceNaNConverter() override {}
};
REGISTER_CONVERTER(ReplaceNaN, ReplaceNaNConverter);

class ConcatAddMulReplaceNaNClipConverter : public Converter {
  std::unique_ptr<repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::ConcatAddMulReplaceNaNClip>();
    auto argMap = getArgumentsFromOperator(op);

    auto c = dyn_cast<repr::ConcatAddMulReplaceNaNClip>(nnOp.get());
    if (argMap.count("clip_min")) {
      CAFFE_ENFORCE(argMap["clip_min"].has_f(), "Invalid 'clip_min' argument");
      c->setClipMin(static_cast<float>(argMap["clip_min"].f()));
    }
    if (argMap.count("clip_max")) {
      CAFFE_ENFORCE(argMap["clip_max"].has_f(), "Invalid 'clip_max' argument");
      c->setClipMin(static_cast<float>(argMap["clip_max"].f()));
    }
    return nnOp;
  }
  OperatorDef convertToOperatorDef(
      const nom::repr::NeuralNetOperator* nnOp) override {
    auto cc_amrc = dyn_cast<repr::ConcatAddMulReplaceNaNClip>(nnOp);
    OperatorDef op;
    op.set_type("ConcatAddMulReplaceNaNClip");
    auto min_arg = op.add_arg();
    min_arg->set_name("clip_min");
    min_arg->set_f(cc_amrc->getClipMin());
    auto max_arg = op.add_arg();
    max_arg->set_name("clip_max");
    max_arg->set_f(cc_amrc->getClipMax());
    op.mutable_device_option()->CopyFrom(getDeviceOption(nnOp));
    return op;
  }
  ~ConcatAddMulReplaceNaNClipConverter() override {}
};
REGISTER_CONVERTER(
    ConcatAddMulReplaceNaNClip,
    ConcatAddMulReplaceNaNClipConverter);

class SliceConverter : public Converter {
  std::unique_ptr<repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::Slice>();
    const caffe2::ArgumentHelper args(op);

    auto c = dyn_cast<repr::Slice>(nnOp.get());
    if (args.HasArgument("starts")) {
      c->setStarts(args.GetRepeatedArgument<int64_t>("starts"));
    }
    if (args.HasArgument("ends")) {
      c->setEnds(args.GetRepeatedArgument<int64_t>("ends"));
    }
    return nnOp;
  }

  OperatorDef convertToOperatorDef(
      const nom::repr::NeuralNetOperator* nnOp) override {
    auto slice = dyn_cast<repr::Slice>(nnOp);
    OperatorDef op;
    op.set_type("Slice");
    op.add_arg()->CopyFrom(
        caffe2::MakeArgument<vector<int64_t>>("starts", slice->getStarts()));
    op.add_arg()->CopyFrom(
        caffe2::MakeArgument<vector<int64_t>>("ends", slice->getEnds()));
    op.mutable_device_option()->CopyFrom(getDeviceOption(nnOp));
    return op;
  }

  ~SliceConverter() override {}
};
REGISTER_CONVERTER(Slice, SliceConverter);

class ClipRangesGatherSigridHashConverter : public Converter {
  std::unique_ptr<repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::ClipRangesGatherSigridHash>();
    const caffe2::ArgumentHelper args(op);

    auto c = dyn_cast<repr::ClipRangesGatherSigridHash>(nnOp.get());
    if (args.HasArgument("feature_indices")) {
      c->setFeatureIndices(
          args.GetRepeatedArgument<int64_t>("feature_indices"));
    }
    if (args.HasArgument("max_lengths")) {
      c->setMaxLengths(args.GetRepeatedArgument<int64_t>("max_lengths"));
    }
    if (args.HasArgument("salts")) {
      c->setSalts(args.GetRepeatedArgument<int64_t>("salts"));
    }
    if (args.HasArgument("max_values")) {
      c->setMaxValues(args.GetRepeatedArgument<int64_t>("max_values"));
    }
    if (args.HasArgument("hash_into_int32")) {
      c->setHashIntoInt32(
          args.GetSingleArgument<bool>("hash_into_int32", false));
    }
    return nnOp;
  }

  OperatorDef convertToOperatorDef(
      const nom::repr::NeuralNetOperator* nnOp) override {
    auto fuse = dyn_cast<repr::ClipRangesGatherSigridHash>(nnOp);
    OperatorDef op;
    op.set_type("ClipRangesGatherSigridHash");
    op.add_arg()->CopyFrom(caffe2::MakeArgument<vector<int64_t>>(
        "feature_indices", fuse->getFeatureIndices()));
    op.add_arg()->CopyFrom(caffe2::MakeArgument<vector<int64_t>>(
        "max_lengths", fuse->getMaxLengths()));
    op.add_arg()->CopyFrom(
        caffe2::MakeArgument<vector<int64_t>>("salts", fuse->getSalts()));
    op.add_arg()->CopyFrom(caffe2::MakeArgument<vector<int64_t>>(
        "max_values", fuse->getMaxValues()));
    op.add_arg()->CopyFrom(caffe2::MakeArgument<bool>(
        "hash_into_int32", fuse->getHashIntoInt32()));
    op.mutable_device_option()->CopyFrom(getDeviceOption(nnOp));
    return op;
  }

  ~ClipRangesGatherSigridHashConverter() override {}
};
REGISTER_CONVERTER(
    ClipRangesGatherSigridHash,
    ClipRangesGatherSigridHashConverter);

class ClipRangesGatherSigridHashV2Converter : public Converter {
  std::unique_ptr<repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::ClipRangesGatherSigridHashV2>();
    const caffe2::ArgumentHelper args(op);

    auto c = dyn_cast<repr::ClipRangesGatherSigridHashV2>(nnOp.get());
    if (args.HasArgument("max_lengths")) {
      c->setMaxLengths(args.GetRepeatedArgument<int64_t>("max_lengths"));
    }
    if (args.HasArgument("salts")) {
      c->setSalts(args.GetRepeatedArgument<int64_t>("salts"));
    }
    if (args.HasArgument("max_values")) {
      c->setMaxValues(args.GetRepeatedArgument<int64_t>("max_values"));
    }
    if (args.HasArgument("hash_into_int32")) {
      c->setHashIntoInt32(
          args.GetSingleArgument<bool>("hash_into_int32", false));
    }
    return nnOp;
  }

  OperatorDef convertToOperatorDef(
      const nom::repr::NeuralNetOperator* nnOp) override {
    auto fuse = dyn_cast<repr::ClipRangesGatherSigridHashV2>(nnOp);
    OperatorDef op;
    op.set_type("ClipRangesGatherSigridHashV2");
    op.add_arg()->CopyFrom(caffe2::MakeArgument<vector<int64_t>>(
        "max_lengths", fuse->getMaxLengths()));
    op.add_arg()->CopyFrom(
        caffe2::MakeArgument<vector<int64_t>>("salts", fuse->getSalts()));
    op.add_arg()->CopyFrom(caffe2::MakeArgument<vector<int64_t>>(
        "max_values", fuse->getMaxValues()));
    op.add_arg()->CopyFrom(caffe2::MakeArgument<bool>(
        "hash_into_int32", fuse->getHashIntoInt32()));
    op.mutable_device_option()->CopyFrom(getDeviceOption(nnOp));
    return op;
  }

  ~ClipRangesGatherSigridHashV2Converter() override {}
};
REGISTER_CONVERTER(
    ClipRangesGatherSigridHashV2,
    ClipRangesGatherSigridHashV2Converter);

class ClipRangesConverter : public Converter {
  std::unique_ptr<repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::ClipRanges>();
    const caffe2::ArgumentHelper args(op);
    auto c = dyn_cast<repr::ClipRanges>(nnOp.get());
    c->setMaxLength(args.GetSingleArgument<int64_t>("max_length", 0));
    return nnOp;
  }

  OperatorDef convertToOperatorDef(
      const nom::repr::NeuralNetOperator* nnOp) override {
    auto clipRanges = dyn_cast<repr::ClipRanges>(nnOp);
    OperatorDef op;
    op.set_type("ClipRanges");
    op.add_arg()->CopyFrom(caffe2::MakeArgument<int64_t>(
        "max_length", clipRanges->getMaxLength()));
    op.mutable_device_option()->CopyFrom(getDeviceOption(nnOp));
    return op;
  }

  ~ClipRangesConverter() override {}
};
REGISTER_CONVERTER(ClipRanges, ClipRangesConverter);

class SigridHashConverter : public Converter {
  std::unique_ptr<repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::SigridHash>();
    const caffe2::ArgumentHelper args(op);
    auto c = dyn_cast<repr::SigridHash>(nnOp.get());
    c->setSalt(args.GetSingleArgument<int64_t>("salt", 0));
    c->setMaxValue(args.GetSingleArgument<int64_t>("maxValue", 0));
    c->setHashIntoInt32(args.GetSingleArgument<bool>("hashIntoInt32", false));
    return nnOp;
  }

  OperatorDef convertToOperatorDef(
      const nom::repr::NeuralNetOperator* nnOp) override {
    auto sigridHash = dyn_cast<repr::SigridHash>(nnOp);
    OperatorDef op;
    op.set_type("SigridHash");
    op.add_arg()->CopyFrom(
        caffe2::MakeArgument<int64_t>("salt", sigridHash->getSalt()));
    op.add_arg()->CopyFrom(
        caffe2::MakeArgument<int64_t>("maxValue", sigridHash->getMaxValue()));
    op.add_arg()->CopyFrom(caffe2::MakeArgument<bool>(
        "hashIntoInt32", sigridHash->getHashIntoInt32()));
    op.mutable_device_option()->CopyFrom(getDeviceOption(nnOp));
    return op;
  }

  ~SigridHashConverter() override {}
};
REGISTER_CONVERTER(SigridHash, SigridHashConverter);

} // namespace
} // namespace caffe2
