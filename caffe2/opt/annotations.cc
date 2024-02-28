#include <utility>

#include "caffe2/opt/annotations.h"

namespace caffe2 {

using namespace nom::repr;

void Caffe2Annotation::setOperatorDef(const caffe2::OperatorDef& opDef) {
  OpDef = opDef;
  OpDefExists = true;
}

bool Caffe2Annotation::hasOperatorDef() const {
  return OpDefExists;
}

const caffe2::OperatorDef& Caffe2Annotation::getOperatorDef() const {
  CAFFE_ENFORCE(
      OpDefExists,
      "OperatorDef was never set.  Use Caffe2Annotation::setOperatorDef.");
  return OpDef;
}
caffe2::OperatorDef* Caffe2Annotation::getMutableOperatorDef() {
  CAFFE_ENFORCE(
      OpDefExists,
      "OperatorDef was never set.  Use Caffe2Annotation::setOperatorDef.");
  return &OpDef;
}

// Distributed annotations
void Caffe2Annotation::setDeviceOption(const caffe2::DeviceOption& devOpt) {
  *OpDef.mutable_device_option() = devOpt;
}

bool Caffe2Annotation::hasDeviceOption() const {
  return OpDef.has_device_option();
}

const caffe2::DeviceOption& Caffe2Annotation::getDeviceOption() const {
  CAFFE_ENFORCE(
      hasDeviceOption(),
      "DeviceOption was never set.  Use Caffe2Annotation::setDeviceOption.");
  return OpDef.device_option();
}
caffe2::DeviceOption* Caffe2Annotation::getMutableDeviceOption() {
  CAFFE_ENFORCE(
      hasDeviceOption(),
      "DeviceOption was never set.  Use Caffe2Annotation::setDeviceOption.");
  return OpDef.mutable_device_option();
}

void Caffe2Annotation::setDevice(std::string device) {
  Device = device;
}
const std::string Caffe2Annotation::getDevice() const {
  return Device;
}

void Caffe2Annotation::setDeviceType(int device) {
  DeviceType = device;
}
int Caffe2Annotation::getDeviceType() const {
  return DeviceType;
}

void Caffe2Annotation::setParallelization(
    Caffe2Annotation::ParallelizationScheme s,
    int num) {
  parallelization_scheme_ = s;
  parallelization_ = num;
}

Caffe2Annotation::ParallelizationScheme
Caffe2Annotation::getParallelizationScheme() const {
  return parallelization_scheme_;
}

int Caffe2Annotation::getParallelization() const {
  return parallelization_;
}

void Caffe2Annotation::setKeyNode(NNGraph::NodeRef n) {
  key_node_ = n;
}
const NNGraph::NodeRef& Caffe2Annotation::getKeyNode() const {
  CAFFE_ENFORCE(key_node_, "No key node has been annotated");
  return key_node_;
}
void Caffe2Annotation::setLengthNode(NNGraph::NodeRef n) {
  length_node_ = n;
}
const NNGraph::NodeRef& Caffe2Annotation::getLengthNode() const {
  CAFFE_ENFORCE(length_node_, "No length node has been annotated");
  return length_node_;
}

void Caffe2Annotation::setComponentLevels(std::vector<std::string> components) {
  component_levels_ = std::move(components);
}
std::vector<std::string> Caffe2Annotation::getComponentLevels() const {
  return component_levels_;
}

bool Caffe2Annotation::classof(const Annotation* A) {
  return A->getKind() == AnnotationKind::Caffe2;
}

} // namespace caffe2
