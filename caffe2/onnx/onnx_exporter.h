/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2.pb.h"
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
using ConvertedResult =
    std::pair<std::vector<NodeProto>, std::vector<TensorProto>>;
} // namespace

class OnnxExporter {
  using SpecialOpConverter = ConvertedResult (OnnxExporter::*)(
      const caffe2::OperatorDef&,
      const std::unordered_map<std::string, caffe2::TensorShape>&);

 public:
  ConvertedResult Caffe2OpToOnnxNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

 private:
  ConvertedResult CommonCaffe2OpToOnnxNodes(const caffe2::OperatorDef& def);

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

  ConvertedResult CreateConcatNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  ConvertedResult CreateLrnNodes(
      const caffe2::OperatorDef& def,
      const std::unordered_map<std::string, caffe2::TensorShape>& shapes);

  // \brief Check black listed arguemnts where we won't pass down when
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
};
} // namespace onnx
} // namespace caffe2
