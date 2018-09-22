#include "caffe2/opt/annotations.h"

namespace caffe2 {

using namespace nom::repr;

CAFFE2_EXPORT void Caffe2Annotation::setOperatorDef(const caffe2::OperatorDef& opDef) {
  OpDef = opDef;
  OpDefExists = true;
}

CAFFE2_EXPORT bool Caffe2Annotation::hasOperatorDef() const {
  return OpDefExists;
}

CAFFE2_EXPORT const caffe2::OperatorDef& Caffe2Annotation::getOperatorDef() const {
  CAFFE_ENFORCE(
      OpDefExists,
      "OperatorDef was never set.  Use Caffe2Annotation::setOperatorDef.");
  return OpDef;
}
CAFFE2_EXPORT caffe2::OperatorDef* Caffe2Annotation::getMutableOperatorDef() {
  CAFFE_ENFORCE(
      OpDefExists,
      "OperatorDef was never set.  Use Caffe2Annotation::setOperatorDef.");
  return &OpDef;
}

// Distributed annotations
CAFFE2_EXPORT void Caffe2Annotation::setDevice(std::string device) {
  Device = device;
}
CAFFE2_EXPORT const std::string Caffe2Annotation::getDevice() const {
  return Device;
}

CAFFE2_EXPORT void Caffe2Annotation::setDeviceType(int device) {
  DeviceType = device;
}
CAFFE2_EXPORT int Caffe2Annotation::getDeviceType() const {
  return DeviceType;
}

CAFFE2_EXPORT void Caffe2Annotation::setParallelization(
    Caffe2Annotation::ParallelizationScheme s,
    int num) {
  parallelization_scheme_ = s;
  parallelization_ = num;
}

CAFFE2_EXPORT Caffe2Annotation::ParallelizationScheme
Caffe2Annotation::getParallelizationScheme() const {
  return parallelization_scheme_;
}

CAFFE2_EXPORT int Caffe2Annotation::getParallelization() const {
  return parallelization_;
}

CAFFE2_EXPORT void Caffe2Annotation::setKeyNode(NNGraph::NodeRef n) {
  key_node_ = n;
}
CAFFE2_EXPORT const NNGraph::NodeRef& Caffe2Annotation::getKeyNode() const {
  CAFFE_ENFORCE(key_node_, "No key node has been annotated");
  return key_node_;
}
CAFFE2_EXPORT void Caffe2Annotation::setLengthNode(NNGraph::NodeRef n) {
  length_node_ = n;
}
CAFFE2_EXPORT const NNGraph::NodeRef& Caffe2Annotation::getLengthNode() const {
  CAFFE_ENFORCE(length_node_, "No length node has been annotated");
  return length_node_;
}

CAFFE2_EXPORT void Caffe2Annotation::setComponentLevels(std::vector<std::string> components) {
  component_levels_ = components;
}
CAFFE2_EXPORT std::vector<std::string> Caffe2Annotation::getComponentLevels() const {
  return component_levels_;
}

CAFFE2_EXPORT bool Caffe2Annotation::classof(const Annotation* A) {
  return A->getKind() == AnnotationKind::Caffe2;
}

} // namespace caffe2
