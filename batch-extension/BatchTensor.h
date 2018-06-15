#pragma once
#include <torch/torch.h>
#include "BatchTensorType.h"
#include <iostream>
#include <vector>

struct BatchTensorType;

struct BatchTensor : public at::TensorImpl {
public:
  BatchTensor();
  BatchTensor(at::Tensor data, at::Tensor mask, std::vector<bool> dims);
  BatchTensor(std::vector<at::Tensor> datalist, std::vector<bool> dims);
  virtual ~BatchTensor(){};
  virtual const char * toString() const override;
  virtual at::IntList sizes() const override {
    throw std::runtime_error("sizes() on BatchTensor");
  }
  virtual at::IntList strides() const override {
    throw std::runtime_error("strides() on BatchTensor");
  }
  virtual int64_t dim() const override {
    throw std::runtime_error("dim() on BatchTensor");
  }
  virtual at::Scalar localScalar() override {
    throw std::runtime_error("localScalar() on BatchTensor");
  }
  virtual void * unsafeGetTH(bool retain) override {
    throw std::runtime_error("unsafeGetTH() on BatchTensor");
  }
  virtual std::unique_ptr<at::Storage> storage() override {
    throw std::runtime_error("storage() on BatchTensor");
  }
  static const char * typeString();
  std::vector<at::Tensor> examples() const;
public:
  at::Tensor data;
  at::Tensor mask;
  std::vector<bool> dims;
  // friend struct BatchTensorType;
};
