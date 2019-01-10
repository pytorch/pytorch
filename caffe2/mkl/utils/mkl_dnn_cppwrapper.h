// Do not directl include this file. Include caffe2/mkl/mkl_utils.h instead.
#ifndef CAFFE2_UTILS_MKL_MKL_DNN_CPPWRAPPER_H
#define CAFFE2_UTILS_MKL_MKL_DNN_CPPWRAPPER_H

#include <stdarg.h>
#include <stddef.h>

#include <mkl.h>

#define C2_MKL_TEMPLATE_PREFIX \
  template <typename T>        \
  inline
#define C2_MKL_SPEC_PREFIX \
  template <>              \
  inline

namespace caffe2 {
namespace mkl {

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnLayoutCreate(
    dnnLayout_t* pLayout,
    size_t dimension,
    const size_t size[],
    const size_t strides[]);
C2_MKL_SPEC_PREFIX dnnError_t dnnLayoutCreate<float>(
    dnnLayout_t* pLayout,
    size_t dimension,
    const size_t size[],
    const size_t strides[]) {
  return dnnLayoutCreate_F32(pLayout, dimension, size, strides);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnLayoutCreate<double>(
    dnnLayout_t* pLayout,
    size_t dimension,
    const size_t size[],
    const size_t strides[]) {
  return dnnLayoutCreate_F64(pLayout, dimension, size, strides);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnLayoutCreateFromPrimitive(
    dnnLayout_t* pLayout,
    const dnnPrimitive_t primitive,
    dnnResourceType_t type);
C2_MKL_SPEC_PREFIX dnnError_t dnnLayoutCreateFromPrimitive<float>(
    dnnLayout_t* pLayout,
    const dnnPrimitive_t primitive,
    dnnResourceType_t type) {
  return dnnLayoutCreateFromPrimitive_F32(pLayout, primitive, type);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnLayoutCreateFromPrimitive<double>(
    dnnLayout_t* pLayout,
    const dnnPrimitive_t primitive,
    dnnResourceType_t type) {
  return dnnLayoutCreateFromPrimitive_F64(pLayout, primitive, type);
}

C2_MKL_TEMPLATE_PREFIX size_t dnnLayoutGetMemorySize(const dnnLayout_t layout);
C2_MKL_SPEC_PREFIX size_t
dnnLayoutGetMemorySize<float>(const dnnLayout_t layout) {
  return dnnLayoutGetMemorySize_F32(layout);
}
C2_MKL_SPEC_PREFIX size_t
dnnLayoutGetMemorySize<double>(const dnnLayout_t layout) {
  return dnnLayoutGetMemorySize_F64(layout);
}

C2_MKL_TEMPLATE_PREFIX int dnnLayoutCompare(
    const dnnLayout_t l1,
    const dnnLayout_t l2);
C2_MKL_SPEC_PREFIX int dnnLayoutCompare<float>(
    const dnnLayout_t l1,
    const dnnLayout_t l2) {
  return dnnLayoutCompare_F32(l1, l2);
}
C2_MKL_SPEC_PREFIX int dnnLayoutCompare<double>(
    const dnnLayout_t l1,
    const dnnLayout_t l2) {
  return dnnLayoutCompare_F64(l1, l2);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t
dnnAllocateBuffer(void** pPtr, dnnLayout_t layout);
C2_MKL_SPEC_PREFIX dnnError_t
dnnAllocateBuffer<float>(void** pPtr, dnnLayout_t layout) {
  return dnnAllocateBuffer_F32(pPtr, layout);
}
C2_MKL_SPEC_PREFIX dnnError_t
dnnAllocateBuffer<double>(void** pPtr, dnnLayout_t layout) {
  return dnnAllocateBuffer_F64(pPtr, layout);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnReleaseBuffer(void* ptr);
C2_MKL_SPEC_PREFIX dnnError_t dnnReleaseBuffer<float>(void* ptr) {
  return dnnReleaseBuffer_F32(ptr);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnReleaseBuffer<double>(void* ptr) {
  return dnnReleaseBuffer_F64(ptr);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnLayoutDelete(dnnLayout_t layout);
C2_MKL_SPEC_PREFIX dnnError_t dnnLayoutDelete<float>(dnnLayout_t layout) {
  return dnnLayoutDelete_F32(layout);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnLayoutDelete<double>(dnnLayout_t layout) {
  return dnnLayoutDelete_F64(layout);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t
dnnPrimitiveAttributesCreate(dnnPrimitiveAttributes_t* attributes);
C2_MKL_SPEC_PREFIX dnnError_t
dnnPrimitiveAttributesCreate<float>(dnnPrimitiveAttributes_t* attributes) {
  return dnnPrimitiveAttributesCreate_F32(attributes);
}
C2_MKL_SPEC_PREFIX dnnError_t
dnnPrimitiveAttributesCreate<double>(dnnPrimitiveAttributes_t* attributes) {
  return dnnPrimitiveAttributesCreate_F64(attributes);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t
dnnPrimitiveAttributesDestroy(dnnPrimitiveAttributes_t attributes);
C2_MKL_SPEC_PREFIX dnnError_t
dnnPrimitiveAttributesDestroy<float>(dnnPrimitiveAttributes_t attributes) {
  return dnnPrimitiveAttributesDestroy_F32(attributes);
}
C2_MKL_SPEC_PREFIX dnnError_t
dnnPrimitiveAttributesDestroy<double>(dnnPrimitiveAttributes_t attributes) {
  return dnnPrimitiveAttributesDestroy_F64(attributes);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnPrimitiveGetAttributes(
    dnnPrimitive_t primitive,
    dnnPrimitiveAttributes_t* attributes);
C2_MKL_SPEC_PREFIX dnnError_t dnnPrimitiveGetAttributes<float>(
    dnnPrimitive_t primitive,
    dnnPrimitiveAttributes_t* attributes) {
  return dnnPrimitiveGetAttributes_F32(primitive, attributes);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnPrimitiveGetAttributes<double>(
    dnnPrimitive_t primitive,
    dnnPrimitiveAttributes_t* attributes) {
  return dnnPrimitiveGetAttributes_F64(primitive, attributes);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t
dnnExecute(dnnPrimitive_t primitive, void* resources[]);
C2_MKL_SPEC_PREFIX dnnError_t
dnnExecute<float>(dnnPrimitive_t primitive, void* resources[]) {
  return dnnExecute_F32(primitive, resources);
}
C2_MKL_SPEC_PREFIX dnnError_t
dnnExecute<double>(dnnPrimitive_t primitive, void* resources[]) {
  return dnnExecute_F64(primitive, resources);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t
dnnExecuteAsync(dnnPrimitive_t primitive, void* resources[]);
C2_MKL_SPEC_PREFIX dnnError_t
dnnExecuteAsync<float>(dnnPrimitive_t primitive, void* resources[]) {
  return dnnExecuteAsync_F32(primitive, resources);
}
C2_MKL_SPEC_PREFIX dnnError_t
dnnExecuteAsync<double>(dnnPrimitive_t primitive, void* resources[]) {
  return dnnExecuteAsync_F64(primitive, resources);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnWaitFor(dnnPrimitive_t primitive);
C2_MKL_SPEC_PREFIX dnnError_t dnnWaitFor<float>(dnnPrimitive_t primitive) {
  return dnnWaitFor_F32(primitive);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnWaitFor<double>(dnnPrimitive_t primitive) {
  return dnnWaitFor_F64(primitive);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnDelete(dnnPrimitive_t primitive);
C2_MKL_SPEC_PREFIX dnnError_t dnnDelete<float>(dnnPrimitive_t primitive) {
  return dnnDelete_F32(primitive);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnDelete<double>(dnnPrimitive_t primitive) {
  return dnnDelete_F64(primitive);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnConversionCreate(
    dnnPrimitive_t* pConversion,
    const dnnLayout_t from,
    const dnnLayout_t to);
C2_MKL_SPEC_PREFIX dnnError_t dnnConversionCreate<float>(
    dnnPrimitive_t* pConversion,
    const dnnLayout_t from,
    const dnnLayout_t to) {
  return dnnConversionCreate_F32(pConversion, from, to);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnConversionCreate<double>(
    dnnPrimitive_t* pConversion,
    const dnnLayout_t from,
    const dnnLayout_t to) {
  return dnnConversionCreate_F64(pConversion, from, to);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t
dnnConversionExecute(dnnPrimitive_t conversion, void* from, void* to);
C2_MKL_SPEC_PREFIX dnnError_t
dnnConversionExecute<float>(dnnPrimitive_t conversion, void* from, void* to) {
  return dnnConversionExecute_F32(conversion, from, to);
}
C2_MKL_SPEC_PREFIX dnnError_t
dnnConversionExecute<double>(dnnPrimitive_t conversion, void* from, void* to) {
  return dnnConversionExecute_F64(conversion, from, to);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnConvolutionCreateForward(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);
C2_MKL_SPEC_PREFIX dnnError_t dnnConvolutionCreateForward<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnConvolutionCreateForward_F32(
      pConvolution,
      attributes,
      algorithm,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}

C2_MKL_SPEC_PREFIX dnnError_t dnnConvolutionCreateForward<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnConvolutionCreateForward_F64(
      pConvolution,
      attributes,
      algorithm,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnConvolutionCreateForwardBias(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);
C2_MKL_SPEC_PREFIX dnnError_t dnnConvolutionCreateForwardBias<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnConvolutionCreateForwardBias_F32(
      pConvolution,
      attributes,
      algorithm,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnConvolutionCreateForwardBias<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnConvolutionCreateForwardBias_F64(
      pConvolution,
      attributes,
      algorithm,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnConvolutionCreateBackwardData(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);
C2_MKL_SPEC_PREFIX dnnError_t dnnConvolutionCreateBackwardData<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnConvolutionCreateBackwardData_F32(
      pConvolution,
      attributes,
      algorithm,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnConvolutionCreateBackwardData<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnConvolutionCreateBackwardData_F64(
      pConvolution,
      attributes,
      algorithm,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnConvolutionCreateBackwardFilter(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);
C2_MKL_SPEC_PREFIX dnnError_t dnnConvolutionCreateBackwardFilter<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnConvolutionCreateBackwardFilter_F32(
      pConvolution,
      attributes,
      algorithm,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnConvolutionCreateBackwardFilter<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnConvolutionCreateBackwardFilter_F64(
      pConvolution,
      attributes,
      algorithm,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnConvolutionCreateBackwardBias(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t dstSize[]);
C2_MKL_SPEC_PREFIX dnnError_t dnnConvolutionCreateBackwardBias<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t dstSize[]) {
  return dnnConvolutionCreateBackwardBias_F32(
      pConvolution, attributes, algorithm, dimension, dstSize);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnConvolutionCreateBackwardBias<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t dstSize[]) {
  return dnnConvolutionCreateBackwardBias_F64(
      pConvolution, attributes, algorithm, dimension, dstSize);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnGroupsConvolutionCreateForward(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);
C2_MKL_SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateForward<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnGroupsConvolutionCreateForward_F32(
      pConvolution,
      attributes,
      algorithm,
      groups,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateForward<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnGroupsConvolutionCreateForward_F64(
      pConvolution,
      attributes,
      algorithm,
      groups,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnGroupsConvolutionCreateForwardBias(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);
C2_MKL_SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateForwardBias<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnGroupsConvolutionCreateForwardBias_F32(
      pConvolution,
      attributes,
      algorithm,
      groups,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateForwardBias<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnGroupsConvolutionCreateForwardBias_F64(
      pConvolution,
      attributes,
      algorithm,
      groups,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardData(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);
C2_MKL_SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardData<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnGroupsConvolutionCreateBackwardData_F32(
      pConvolution,
      attributes,
      algorithm,
      groups,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardData<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnGroupsConvolutionCreateBackwardData_F64(
      pConvolution,
      attributes,
      algorithm,
      groups,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardFilter(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);
C2_MKL_SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardFilter<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnGroupsConvolutionCreateBackwardFilter_F32(
      pConvolution,
      attributes,
      algorithm,
      groups,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardFilter<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnGroupsConvolutionCreateBackwardFilter_F64(
      pConvolution,
      attributes,
      algorithm,
      groups,
      dimension,
      srcSize,
      dstSize,
      filterSize,
      convolutionStrides,
      inputOffset,
      border_type);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardBias(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t dstSize[]);
C2_MKL_SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardBias<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t dstSize[]) {
  return dnnGroupsConvolutionCreateBackwardBias_F32(
      pConvolution, attributes, algorithm, groups, dimension, dstSize);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardBias<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t dstSize[]) {
  return dnnGroupsConvolutionCreateBackwardBias_F64(
      pConvolution, attributes, algorithm, groups, dimension, dstSize);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnReLUCreateForward(
    dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float negativeSlope);
C2_MKL_SPEC_PREFIX dnnError_t dnnReLUCreateForward<float>(
    dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float negativeSlope) {
  return dnnReLUCreateForward_F32(pRelu, attributes, dataLayout, negativeSlope);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnReLUCreateForward<double>(
    dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float negativeSlope) {
  return dnnReLUCreateForward_F64(pRelu, attributes, dataLayout, negativeSlope);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnReLUCreateBackward(
    dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout,
    float negativeSlope);
C2_MKL_SPEC_PREFIX dnnError_t dnnReLUCreateBackward<float>(
    dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout,
    float negativeSlope) {
  return dnnReLUCreateBackward_F32(
      pRelu, attributes, diffLayout, dataLayout, negativeSlope);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnReLUCreateBackward<double>(
    dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout,
    float negativeSlope) {
  return dnnReLUCreateBackward_F64(
      pRelu, attributes, diffLayout, dataLayout, negativeSlope);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnLRNCreateForward(
    dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    size_t kernel_size,
    float alpha,
    float beta,
    float k);
C2_MKL_SPEC_PREFIX dnnError_t dnnLRNCreateForward<float>(
    dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    size_t kernel_size,
    float alpha,
    float beta,
    float k) {
  return dnnLRNCreateForward_F32(
      pLrn, attributes, dataLayout, kernel_size, alpha, beta, k);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnLRNCreateForward<double>(
    dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    size_t kernel_size,
    float alpha,
    float beta,
    float k) {
  return dnnLRNCreateForward_F64(
      pLrn, attributes, dataLayout, kernel_size, alpha, beta, k);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnLRNCreateBackward(
    dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout,
    size_t kernel_size,
    float alpha,
    float beta,
    float k);
C2_MKL_SPEC_PREFIX dnnError_t dnnLRNCreateBackward<float>(
    dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout,
    size_t kernel_size,
    float alpha,
    float beta,
    float k) {
  return dnnLRNCreateBackward_F32(
      pLrn, attributes, diffLayout, dataLayout, kernel_size, alpha, beta, k);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnLRNCreateBackward<double>(
    dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout,
    size_t kernel_size,
    float alpha,
    float beta,
    float k) {
  return dnnLRNCreateBackward_F64(
      pLrn, attributes, diffLayout, dataLayout, kernel_size, alpha, beta, k);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnPoolingCreateForward(
    dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op,
    const dnnLayout_t srcLayout,
    const size_t kernelSize[],
    const size_t kernelStride[],
    const int inputOffset[],
    const dnnBorder_t border_type);
C2_MKL_SPEC_PREFIX dnnError_t dnnPoolingCreateForward<float>(
    dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op,
    const dnnLayout_t srcLayout,
    const size_t kernelSize[],
    const size_t kernelStride[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnPoolingCreateForward_F32(
      pPooling,
      attributes,
      op,
      srcLayout,
      kernelSize,
      kernelStride,
      inputOffset,
      border_type);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnPoolingCreateForward<double>(
    dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op,
    const dnnLayout_t srcLayout,
    const size_t kernelSize[],
    const size_t kernelStride[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnPoolingCreateForward_F64(
      pPooling,
      attributes,
      op,
      srcLayout,
      kernelSize,
      kernelStride,
      inputOffset,
      border_type);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnPoolingCreateBackward(
    dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op,
    const dnnLayout_t srcLayout,
    const size_t kernelSize[],
    const size_t kernelStride[],
    const int inputOffset[],
    const dnnBorder_t border_type);
C2_MKL_SPEC_PREFIX dnnError_t dnnPoolingCreateBackward<float>(
    dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op,
    const dnnLayout_t srcLayout,
    const size_t kernelSize[],
    const size_t kernelStride[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnPoolingCreateBackward_F32(
      pPooling,
      attributes,
      op,
      srcLayout,
      kernelSize,
      kernelStride,
      inputOffset,
      border_type);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnPoolingCreateBackward<double>(
    dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op,
    const dnnLayout_t srcLayout,
    const size_t kernelSize[],
    const size_t kernelStride[],
    const int inputOffset[],
    const dnnBorder_t border_type) {
  return dnnPoolingCreateBackward_F64(
      pPooling,
      attributes,
      op,
      srcLayout,
      kernelSize,
      kernelStride,
      inputOffset,
      border_type);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnConcatCreate(
    dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes,
    const size_t N,
    dnnLayout_t src[]);
C2_MKL_SPEC_PREFIX dnnError_t dnnConcatCreate<float>(
    dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes,
    const size_t N,
    dnnLayout_t src[]) {
  return dnnConcatCreate_F32(pConcat, attributes, N, src);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnConcatCreate<double>(
    dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes,
    const size_t N,
    dnnLayout_t src[]) {
  return dnnConcatCreate_F64(pConcat, attributes, N, src);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnSplitCreate(
    dnnPrimitive_t* pSplit,
    dnnPrimitiveAttributes_t attributes,
    const size_t N,
    dnnLayout_t src,
    size_t dst[]);
C2_MKL_SPEC_PREFIX dnnError_t dnnSplitCreate<float>(
    dnnPrimitive_t* pSplit,
    dnnPrimitiveAttributes_t attributes,
    const size_t N,
    dnnLayout_t src,
    size_t dst[]) {
  return dnnSplitCreate_F32(pSplit, attributes, N, src, dst);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnSplitCreate<double>(
    dnnPrimitive_t* pSplit,
    dnnPrimitiveAttributes_t attributes,
    const size_t N,
    dnnLayout_t src,
    size_t dst[]) {
  return dnnSplitCreate_F64(pSplit, attributes, N, src, dst);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnSumCreate(
    dnnPrimitive_t* pSum,
    dnnPrimitiveAttributes_t attributes,
    const size_t nSummands,
    dnnLayout_t layout,
    T* coefficients);
C2_MKL_SPEC_PREFIX dnnError_t dnnSumCreate<float>(
    dnnPrimitive_t* pSum,
    dnnPrimitiveAttributes_t attributes,
    const size_t nSummands,
    dnnLayout_t layout,
    float* coefficients) {
  return dnnSumCreate_F32(pSum, attributes, nSummands, layout, coefficients);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnSumCreate<double>(
    dnnPrimitive_t* pSum,
    dnnPrimitiveAttributes_t attributes,
    const size_t nSummands,
    dnnLayout_t layout,
    double* coefficients) {
  return dnnSumCreate_F64(pSum, attributes, nSummands, layout, coefficients);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnBatchNormalizationCreateForward(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps);
C2_MKL_SPEC_PREFIX dnnError_t dnnBatchNormalizationCreateForward<float>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps) {
  return dnnBatchNormalizationCreateForward_F32(
      pBatchNormalization, attributes, dataLayout, eps);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnBatchNormalizationCreateForward<double>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps) {
  return dnnBatchNormalizationCreateForward_F64(
      pBatchNormalization, attributes, dataLayout, eps);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnBatchNormalizationCreateBackwardData(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps);
C2_MKL_SPEC_PREFIX dnnError_t dnnBatchNormalizationCreateBackwardData<float>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps) {
  return dnnBatchNormalizationCreateBackwardData_F32(
      pBatchNormalization, attributes, dataLayout, eps);
}

C2_MKL_SPEC_PREFIX dnnError_t dnnBatchNormalizationCreateBackwardData<double>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps) {
  return dnnBatchNormalizationCreateBackwardData_F64(
      pBatchNormalization, attributes, dataLayout, eps);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnBatchNormalizationCreateBackwardScaleShift(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps);
C2_MKL_SPEC_PREFIX dnnError_t
dnnBatchNormalizationCreateBackwardScaleShift<float>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps) {
  return dnnBatchNormalizationCreateBackwardScaleShift_F32(
      pBatchNormalization, attributes, dataLayout, eps);
}
C2_MKL_SPEC_PREFIX dnnError_t
dnnBatchNormalizationCreateBackwardScaleShift<double>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps) {
  return dnnBatchNormalizationCreateBackwardScaleShift_F64(
      pBatchNormalization, attributes, dataLayout, eps);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnBatchNormalizationCreateForward_v2(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps,
    unsigned int flags);
C2_MKL_SPEC_PREFIX dnnError_t dnnBatchNormalizationCreateForward_v2<float>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps,
    unsigned int flags) {
  return dnnBatchNormalizationCreateForward_v2_F32(
      pBatchNormalization, attributes, dataLayout, eps, flags);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnBatchNormalizationCreateForward_v2<double>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps,
    unsigned int flags) {
  return dnnBatchNormalizationCreateForward_v2_F64(
      pBatchNormalization, attributes, dataLayout, eps, flags);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnBatchNormalizationCreateBackward_v2(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps,
    unsigned int flags);
C2_MKL_SPEC_PREFIX dnnError_t dnnBatchNormalizationCreateBackward_v2<float>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps,
    unsigned int flags) {
  return dnnBatchNormalizationCreateBackward_v2_F32(
      pBatchNormalization, attributes, dataLayout, eps, flags);
}

C2_MKL_SPEC_PREFIX dnnError_t dnnBatchNormalizationCreateBackward_v2<double>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps,
    unsigned int flags) {
  return dnnBatchNormalizationCreateBackward_v2_F64(
      pBatchNormalization, attributes, dataLayout, eps, flags);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnInnerProductCreateForward(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels);
C2_MKL_SPEC_PREFIX dnnError_t dnnInnerProductCreateForward<float>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels) {
  return dnnInnerProductCreateForward_F32(
      pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnInnerProductCreateForward<double>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels) {
  return dnnInnerProductCreateForward_F64(
      pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnInnerProductCreateForwardBias(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels);
C2_MKL_SPEC_PREFIX dnnError_t dnnInnerProductCreateForwardBias<float>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels) {
  return dnnInnerProductCreateForwardBias_F32(
      pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnInnerProductCreateForwardBias<double>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels) {
  return dnnInnerProductCreateForwardBias_F64(
      pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnInnerProductCreateBackwardData(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels);
C2_MKL_SPEC_PREFIX dnnError_t dnnInnerProductCreateBackwardData<float>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels) {
  return dnnInnerProductCreateBackwardData_F32(
      pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnInnerProductCreateBackwardData<double>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels) {
  return dnnInnerProductCreateBackwardData_F64(
      pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnInnerProductCreateBackwardFilter(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels);
C2_MKL_SPEC_PREFIX dnnError_t dnnInnerProductCreateBackwardFilter<float>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels) {
  return dnnInnerProductCreateBackwardFilter_F32(
      pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnInnerProductCreateBackwardFilter<double>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels) {
  return dnnInnerProductCreateBackwardFilter_F64(
      pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

C2_MKL_TEMPLATE_PREFIX dnnError_t dnnInnerProductCreateBackwardBias(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[]);
C2_MKL_SPEC_PREFIX dnnError_t dnnInnerProductCreateBackwardBias<float>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[]) {
  return dnnInnerProductCreateBackwardBias_F32(
      pInnerProduct, attributes, dimensions, srcSize);
}
C2_MKL_SPEC_PREFIX dnnError_t dnnInnerProductCreateBackwardBias<double>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[]) {
  return dnnInnerProductCreateBackwardBias_F64(
      pInnerProduct, attributes, dimensions, srcSize);
}

} // namespace mkl
} // namespace caffe2

// Undef macros to make sure that things are clean.
#undef C2_MKL_TEMPLATE_PREFIX
#undef C2_MKL_SPEC_PREFIX

#endif // CAFFE2_UTILS_MKL_MKL_DNN_CPPWRAPPER_H
