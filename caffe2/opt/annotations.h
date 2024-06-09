#pragma once

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2_pb.h"
#include "nomnigraph/Representations/NeuralNet.h"

namespace caffe2 {

class TORCH_API Caffe2Annotation : public nom::repr::Annotation {
 public:
  Caffe2Annotation() : Annotation(AnnotationKind::Caffe2) {}
  Caffe2Annotation(std::string device)
      : Annotation(AnnotationKind::Caffe2), Device(device) {}
  virtual ~Caffe2Annotation() {}

  void setOperatorDef(const caffe2::OperatorDef& opDef);
  bool hasOperatorDef() const;
  const caffe2::OperatorDef& getOperatorDef() const;
  caffe2::OperatorDef* getMutableOperatorDef();

  void setDeviceOption(const caffe2::DeviceOption& opDef);
  bool hasDeviceOption() const;
  const caffe2::DeviceOption& getDeviceOption() const;
  caffe2::DeviceOption* getMutableDeviceOption();

  // Distributed annotations
  void setDevice(std::string device);
  const std::string getDevice() const;
  void setDeviceType(int device);
  int getDeviceType() const;

  enum class ParallelizationScheme {
    none,
    split_by_batch,
    split_by_length,
    shard,
    shard_by_number
  };
  void setParallelization(ParallelizationScheme, int num = -1);
  ParallelizationScheme getParallelizationScheme() const;
  int getParallelization() const;

  void setKeyNode(nom::repr::NNGraph::NodeRef);
  const nom::repr::NNGraph::NodeRef& getKeyNode() const;
  void setLengthNode(nom::repr::NNGraph::NodeRef);
  const nom::repr::NNGraph::NodeRef& getLengthNode() const;

  void setComponentLevels(std::vector<std::string> components);
  std::vector<std::string> getComponentLevels() const;

  static bool classof(const Annotation* A);

 private:
  std::string Device = "";
  caffe2::OperatorDef OpDef;
  bool OpDefExists = false;

  // Distributed annotations
  int DeviceType = caffe2::DeviceTypeProto::PROTO_CPU;
  ParallelizationScheme parallelization_scheme_ = ParallelizationScheme::none;
  int parallelization_ = -1;
  nom::repr::NNGraph::NodeRef key_node_ = nullptr;
  nom::repr::NNGraph::NodeRef length_node_ = nullptr;
  std::vector<std::string> component_levels_;
};

} // namespace caffe2
