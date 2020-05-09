#pragma once

#include "caffe2/core/common.h"
#include "caffe2/core/tensor.h"
#include "caffe2/onnx/helper.h"
#include "caffe2/proto/caffe2_pb.h"
#include "onnx/onnx_pb.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace caffe2 {
namespace onnx {

namespace {
using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::GraphProto;
using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::NodeProto;
using ::ONNX_NAMESPACE::TensorProto;
} // namespace

using ConvertedResult =
    std::pair<std::vector<NodeProto>, std::vector<TensorProto>>;

// Rewrite Caffe2 nets into SSA forms. Notice that we will preserve the external
// output names for predict net.
CAFFE2_API std::unordered_map<std::string, std::string> SsaRewrite(
    caffe2::NetDef* init_net,
    caffe2::NetDef* pred_net);

::ONNX_NAMESPACE::TensorProto::DataType Caffe2TypeToOnnxType(
    caffe2::TensorProto::DataType t);

class CAFFE2_API OnnxExporter {
  using SpecialOpConverter = ConvertedResult (OnnxExporter::*)(
      const caffe2::OperatorDef&,
      const std::unordered_map<std::string, caffe2::TensorShape>&);

 public:
  OnnxExporter(DummyName* dummy = nullptr) {
    if (dummy) {
      dummy_ = std::shared_ptr<DummyName>(dummy, [](DummyName*) {});
    } else {
      dummy_ = std::make_shared<DummyName>();
    }
  }

  ConvertedResult Caffe2OpToOnnxNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  void InitOpToTensorProto(const caffe2::OperatorDef& def, TensorProto* tensor);
 private:
  ConvertedResult CommonCaffe2OpToOnnxNodes(const caffe2::OperatorDef& def);

  ConvertedResult CreateArgMaxMinOpNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateBinaryElementwiseOpNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateCastNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateElementwiseLinearNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateConvPoolNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateGemmNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateReshapeNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateSliceNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateChannelShuffleNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateReduceMeanNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateConcatNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateMergeDimNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateLrnNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateUpsampleNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  // \brief Check black listed arguments where we won't pass down when
  // converting to ONNX node
  bool IsBlackListed(const caffe2::Argument& arg);

  // \brief Convert Caffe2 argument to Onnx attribute
  void CopyCaffe2ArgToOnnxAttr(
      AttributeProto* attr,
      const std::string& op_type,
      const caffe2::Argument& arg);

  // LUT getters
  const std::unordered_map<std::string, std::string>& get_renamed_operators()
      const;
  const std::unordered_map<std::string, std::string>& get_renamed_attrs() const;
  const std::
      unordered_map<std::string, std::unordered_map<std::string, std::string>>&
      get_per_op_renamed_attrs() const;
  const std::unordered_map<std::string, OnnxExporter::SpecialOpConverter>&
  get_special_operators() const;

  // Dummy name generator
  std::shared_ptr<DummyName> dummy_;
};
} // namespace onnx
} // namespace caffe2
