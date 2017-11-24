#pragma once

#include "nnpack.h"

#include "ATen/ATen.h"

namespace torch {
namespace nnpack {

void SpatialConvolution_updateOutput(
    at::Tensor& input,
    at::Tensor& output,
    at::Tensor& weight,
    at::Tensor& bias,
    int kW,
    int kH,
    int padW,
    int padH);

void SpatialConvolution_updateGradInput(
    at::Tensor& input,
    at::Tensor& gradOutput,
    at::Tensor& gradInput,
    at::Tensor& weight,
    int kW,
    int kH,
    int padW,
    int padH);

void SpatialConvolution_accGradWeight(
    at::Tensor& input,
    at::Tensor& gradOutput,
    at::Tensor& gradWeight,
    int kW,
    int kH,
    int padW,
    int padH);

} // torch::nnpack
} // torch
