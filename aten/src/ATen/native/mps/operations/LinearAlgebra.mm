//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif


namespace at {
namespace native {

/*
 * Helper functions to be used for mm/addmm for detecting the Transpositions
 * when doing Batched GEMM operations.
 */

static Tensor prepare_batch_matrix_by_transposing(const Tensor& tensor,
                                       bool& transpose_tensor,
                                       int64_t& ld_tensor,
                                       bool transpose_result,
                                       int64_t m, int64_t n) {
  IntArrayRef tensor_strides = tensor.strides();
  Tensor tensor_;
  int fast_dim = transpose_result ? 2 : 1;
  int leading_dim = transpose_result ? 1 : 2;

  if (tensor_strides[fast_dim] == 1 &&
    (tensor_strides[leading_dim] >= std::max<int64_t>(1, m))) {
    transpose_tensor = false;
    tensor_ = tensor;
    ld_tensor = tensor_strides[leading_dim];
  } else if ((tensor_strides[leading_dim] == 1) &&
    (tensor_strides[fast_dim] >= std::max<int64_t>(1, n))) {
    transpose_tensor = true;
    tensor_ = tensor;
    ld_tensor = tensor_strides[fast_dim];
  } else {
    transpose_tensor = !transpose_result;
    // gemm call requires leading dimension and stride parameters to be non-zero
    bool is_stride_non_zero = tensor.stride(1) != 0 && tensor.stride(2) != 0;
    if (tensor.is_contiguous() && is_stride_non_zero) {
      tensor_ = tensor;
    } else {
      tensor_ = tensor.clone(at::MemoryFormat::Contiguous);
    }
    ld_tensor = tensor_.stride(1);
  }

  return tensor_;
}

/*
 * Helper functions to be used for mm/addmm for detecting the Transpositions
 * when doing GEMM operations.
 */
void prepare_matrices_for_broadcasting(
  const Tensor * bias,
  const Tensor & self,
  const Tensor & other,
  const Scalar * beta,
  bool * transpose_mat1_times_mat2,
  bool & transpose_mat1,
  bool & transpose_mat2) {
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "tensors must be 2-D");
  if (bias && beta->toDouble() != 0.0f) {
    TORCH_CHECK(bias->dim() == 2, "tensors must be 2-D");
  }

  std::pair<int64_t, int64_t> mat1_sizes;
  std::pair<int64_t, int64_t> mat2_sizes;

  mat1_sizes = std::make_pair(self.sizes()[0], self.sizes()[1]);
  mat2_sizes = std::make_pair(other.sizes()[0], other.sizes()[1]);

  if (mat1_sizes == mat2_sizes) {
    transpose_mat2 = true;
    std::swap(mat2_sizes.first, mat2_sizes.second);
  }
  if (bias && beta && transpose_mat1_times_mat2) {
    if (beta->toDouble() != 0.0f && mat1_sizes.first == bias->sizes()[1] && mat2_sizes.second == bias->sizes()[0])
      *transpose_mat1_times_mat2 = true;
  }
}

enum LinearAlgebraOpType {
  ADDBMM_OP_TYPE,
  BADDBMM_OP_TYPE
};

Tensor& mm_out_mps_impl(
    const Tensor& self,
    const Tensor& other,
    Tensor& output) {
  using namespace mps;
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(self.scalar_type() == ScalarType::Double
              || self.scalar_type() == ScalarType::Float
              || self.scalar_type() == ScalarType::Half, "MPS device does not support mm for non-float inputs");

  TensorArg args[]{{output, "out", 0}, {self, "mat1", 1}, {other, "mat2", 2}};
  checkAllSameGPU("mm", args);

  TORCH_CHECK(output.is_mps());

  // Transpose inputs if needed
  IntArrayRef output_sizes = output.sizes();
  if ((output_sizes[0] == 0) || (output_sizes[1] == 0)) {
    return output;
  }

  struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *selfTensor_ = nil;
    MPSGraphTensor *otherTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  @autoreleasepool {

    string key = "mm_out_mps_impl" + getTensorsStringKey({self, other});

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {

      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *selfTensor = nil;
          MPSGraphTensor *otherTensor = nil;
          MPSGraphTensor *outputTensor = nil;

          if(self.numel() == 0 || other.numel() == 0) {

            outputTensor = [mpsGraph constantWithScalar:0.
                                                  shape:getMPSShape(output_sizes)
                                                   dataType:getMPSDataType(output.scalar_type())];

          }
          else {

            selfTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, self);
            otherTensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, other);
            outputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:selfTensor
                                                           secondaryTensor:otherTensor
                                                                      name:nil];
          }

          newCachedGraph->selfTensor_ = selfTensor;
          newCachedGraph->otherTensor_ = otherTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }
    Placeholder selfPlaceholder = Placeholder();
    Placeholder otherPlaceholder = Placeholder();
    if(!(self.numel() == 0 || other.numel() == 0)) {
      selfPlaceholder = Placeholder(cachedGraph->selfTensor_, self);
      otherPlaceholder = Placeholder(cachedGraph->otherTensor_, other);
    }
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = nil;

    if(!(self.numel() == 0 || other.numel() == 0))
      feeds = @{
        selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
        otherPlaceholder.getMPSGraphTensor() : otherPlaceholder.getMPSGraphTensorData()
      };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return output;
}

Tensor& addmm_out_mps_impl(
    const Tensor& bias,
    const Tensor& self,  // input
    const Tensor& other, // weight
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& output) {
  using namespace mps;

  TORCH_CHECK(output.is_mps());
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(self.scalar_type() == ScalarType::Double
              || self.scalar_type() == ScalarType::Float
              || self.scalar_type() == ScalarType::Half, "MPS device does not support addmm for non-float input");

  TensorArg args[]{{output, "out", 0}, {bias, "self", 1}, {self, "mat1", 2}, {other, "mat2", 3}};
  checkAllSameGPU(__func__, args);

  IntArrayRef mat1_sizes = self.sizes();
  IntArrayRef mat2_sizes = other.sizes();
  IntArrayRef bias_sizes;
  c10::MaybeOwned<Tensor> bias_;
  if (&output != &bias) {
    bias_ = expand_size(bias, {mat1_sizes[0], mat2_sizes[1]}, "addmm");
    bias_sizes = bias_->sizes();
  } else {
    bias_ = c10::MaybeOwned<Tensor>::borrowed(bias);
    bias_sizes = bias_->sizes();
    TORCH_CHECK(output.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(bias_sizes[0] == mat1_sizes[0], "self_ dim 0 must match mat1 dim 0");
    TORCH_CHECK(bias_sizes[1] == mat2_sizes[1], "self_ dim 1 must match mat2 dim 1");
  }

  if (&output != &self) {
    output.resize_(bias_sizes);
    if (beta.toComplexDouble() != 0.0) {
      at::native::copy_(output, *bias_);
    }
  }
  IntArrayRef output_sizes = output.sizes();
  if ((output_sizes[0] == 0) || (output_sizes[1] == 0)) {
    return output;
  }

  MPSStream* stream = getCurrentMPSStream();

  bool transpose_mat1_times_mat2 = false;
  bool transpose_mat1            = false;
  bool transpose_mat2            = false;

  prepare_matrices_for_broadcasting(&(*bias_), self, other, &beta, &transpose_mat1_times_mat2, transpose_mat1, transpose_mat2);

  struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *selfTensor_ = nil;
    MPSGraphTensor *otherTensor_ = nil;
    MPSGraphTensor *biasTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "addmm_out_mps_impl" + getTensorsStringKey({self, other, *bias_})
                                       + ":" + to_string(transpose_mat1) + ":" + to_string(transpose_mat2)
                                       + ":" + to_string(beta.toDouble())
                                       + ":" + to_string(alpha.toDouble());
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {

      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *selfTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor *otherTensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, other);
          MPSGraphTensor *biasTensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, *bias_);

          MPSGraphTensor* t1 = nil;
          MPSGraphTensor* t2 = nil;

          if(transpose_mat1)
            t1 = [mpsGraph transposeTensor:selfTensor
                                 dimension:-1
                             withDimension:-2
                                      name:nil];
          else
            t1 = selfTensor;

          if(transpose_mat2)
            t2 = [mpsGraph transposeTensor:otherTensor
                                 dimension:-1
                             withDimension:-2
                                      name:nil];
          else
            t2 = otherTensor;


          // TODO: Use alpha and beta here with fill_.Scalar and mul
          // Intermediate as placeholder
          MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:t1
                                                                          secondaryTensor:t2
                                                                                     name:@"MM/(mat1@mat2)"];

          // Intermediates for beta and alpha
          MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar:beta.toDouble()
                                                           dataType:getMPSScalarType((*bias_).scalar_type())];
          MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha.toDouble()
                                                           dataType:getMPSScalarType(self.scalar_type())];

          // Intermediates for multiplying by beta and alpha
          MPSGraphTensor* productTimesAlphaTensor = [mpsGraph multiplicationWithPrimaryTensor:productTensor
                                                                              secondaryTensor:alphaTensor
                                                                                         name:@"MM/alpha*(mat1@mat2)"];
          MPSGraphTensor* biasTimesBetaTensor = [mpsGraph multiplicationWithPrimaryTensor:biasTensor
                                                                          secondaryTensor:betaTensor
                                                                                     name:@"MM/beta*input"];

          if (transpose_mat1_times_mat2)
            biasTimesBetaTensor = [mpsGraph transposeTensor: biasTimesBetaTensor
                                                  dimension: -1
                                              withDimension: -2
                                                       name: nil];

          MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:productTimesAlphaTensor
                                                             secondaryTensor:biasTimesBetaTensor
                                                                        name:@"MM/beta*input + alpha*(mat1@mat2)"];

          newCachedGraph->selfTensor_ = selfTensor;
          newCachedGraph->otherTensor_ = otherTensor;
          newCachedGraph->biasTensor_ = biasTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->selfTensor_, self);
    Placeholder otherPlaceholder = Placeholder(cachedGraph->otherTensor_, other);
    Placeholder biasPlaceholder = Placeholder(cachedGraph->biasTensor_, *bias_);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      otherPlaceholder.getMPSGraphTensor() : otherPlaceholder.getMPSGraphTensorData(),
      biasPlaceholder.getMPSGraphTensor() : biasPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return output;
}


Tensor& bmm_out_mps_impl(
  const Tensor & batch1,
  const Tensor & batch2,
  Tensor & result) {
  using namespace mps;

  TORCH_CHECK(batch1.scalar_type() == ScalarType::Double
              || batch1.scalar_type() == ScalarType::Float
              || batch1.scalar_type() == ScalarType::Half, "MPS device does not support bmm for non-float inputs");

  if (batch1.numel() == 0 || batch2.numel() == 0) {
    return result;
  }

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *batch1Tensor_ = nil;
    MPSGraphTensor *batch2Tensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "bmm_out_mps_impl" + getTensorsStringKey({batch1, batch2});

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {

      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *batch1Tensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, batch1);
          MPSGraphTensor *batch2Tensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, batch2);

          MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:batch1Tensor
                                                                          secondaryTensor:batch2Tensor
                                                                                     name:@"MM/(batch1@batch2)"];

          newCachedGraph->batch1Tensor_ = batch1Tensor;
          newCachedGraph->batch2Tensor_ = batch2Tensor;
          newCachedGraph->outputTensor_ = productTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }
    Placeholder batch1Placeholder = Placeholder(cachedGraph->batch1Tensor_, batch1);
    Placeholder batch2Placeholder = Placeholder(cachedGraph->batch2Tensor_, batch2);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      batch1Placeholder.getMPSGraphTensor() : batch1Placeholder.getMPSGraphTensorData(),
      batch2Placeholder.getMPSGraphTensor() : batch2Placeholder.getMPSGraphTensorData(),
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;
}

Tensor& addbmm_or_baddbmm_out_mps_impl(
  const Tensor       & input,
  const Tensor       & batch1,
  const Tensor       & batch2,
  const Scalar       & beta,
  const Scalar       & alpha,
  Tensor             & result,
  LinearAlgebraOpType  opType) {
  using namespace mps;

  TORCH_CHECK(input.is_mps());
  TORCH_CHECK(batch1.is_mps());
  TORCH_CHECK(batch2.is_mps());
  TORCH_CHECK(result.is_mps());

  TORCH_CHECK(batch1.scalar_type() == ScalarType::Double
              || batch1.scalar_type() == ScalarType::Float
              || batch1.scalar_type() == ScalarType::Half, "MPS device does not support addbmm or baddbmm for non-float inputs");

  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
  TORCH_CHECK(batch1.size(0) == batch2.size(0),
      "batch1 and batch2 must have same number of batches, got ",
      batch1.size(0), " and ", batch2.size(0));
  TORCH_CHECK(batch1.size(2) == batch2.size(1),
      "Incompatible matrix sizes for bmm (",
      batch1.size(1), "x", batch1.size(2), " and ",
      batch2.size(1), "x", batch2.size(2), ")");

  if (opType == ADDBMM_OP_TYPE)
  {
    result.resize_as_(input);

    const int64_t num_batches = batch1.size(0);

    if (num_batches == 0) {
      result.zero_();
      return result;
    }
  }

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *batch1Tensor_ = nil;
    MPSGraphTensor *batch2Tensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = (opType == ADDBMM_OP_TYPE) ? ("addbmm_out_mps_impl") : ("baddbmm_out_mps_impl");
    key += getTensorsStringKey({batch1, batch2, input})
               + ":" + to_string(beta.toDouble())
               + ":" + to_string(alpha.toDouble());

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {

      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *inputTensor  = mps::mpsGraphRankedPlaceHolder(mpsGraph, input);
          MPSGraphTensor *batch1Tensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, batch1);
          MPSGraphTensor *batch2Tensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, batch2);

          // Intermediates for beta and alpha
          MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar: beta.toDouble()
                                                           dataType: getMPSScalarType(input.scalar_type())];
          MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar: alpha.toDouble()
                                                            dataType: getMPSScalarType(batch1.scalar_type())];

          MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:batch1Tensor
                                                                          secondaryTensor:batch2Tensor
                                                                                     name:@"(batch1@batch2)"];

          MPSGraphTensor* reductionSumTensor = productTensor;
          if (opType == ADDBMM_OP_TYPE) {
            reductionSumTensor = [mpsGraph reductionSumWithTensor: productTensor
                                                             axis: 0
                                                             name: @"reductionSum(batch1@batch2)"];
          }

          // Intermediates for multiplying by beta and alpha
          MPSGraphTensor* reductionSumTimesAlphaTensor = [mpsGraph multiplicationWithPrimaryTensor: reductionSumTensor
                                                                              secondaryTensor: alphaTensor
                                                                                         name: @"alpha*(batch1@batch2)"];
          MPSGraphTensor* biasTimesBetaTensor = [mpsGraph multiplicationWithPrimaryTensor: inputTensor
                                                                          secondaryTensor: betaTensor
                                                                                     name: @"beta*input"];

          MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:reductionSumTimesAlphaTensor
                                                             secondaryTensor:biasTimesBetaTensor
                                                                        name:@"beta*input + alpha*(batch1@batch2)"];

          newCachedGraph->inputTensor_  = inputTensor;
          newCachedGraph->batch1Tensor_ = batch1Tensor;
          newCachedGraph->batch2Tensor_ = batch2Tensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }
    Placeholder inputPlaceholder  = Placeholder(cachedGraph->inputTensor_,  input);
    Placeholder batch1Placeholder = Placeholder(cachedGraph->batch1Tensor_, batch1);
    Placeholder batch2Placeholder = Placeholder(cachedGraph->batch2Tensor_, batch2);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      inputPlaceholder.getMPSGraphTensor()  : inputPlaceholder.getMPSGraphTensorData(),
      batch1Placeholder.getMPSGraphTensor() : batch1Placeholder.getMPSGraphTensorData(),
      batch2Placeholder.getMPSGraphTensor() : batch2Placeholder.getMPSGraphTensorData(),
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;
}

TORCH_IMPL_FUNC(mm_out_mps)(const Tensor& self, const Tensor& mat2, const Tensor& result) {
  mm_out_mps_impl(self, mat2, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(addmm_out_mps)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
  addmm_out_mps_impl(self, mat1, mat2, beta, alpha, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(bmm_out_mps) (const Tensor & batch1, const Tensor & batch2, const Tensor & result) {
  bmm_out_mps_impl(batch1, batch2, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(baddbmm_out_mps) (const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
  addbmm_or_baddbmm_out_mps_impl(self, batch1, batch2, beta, alpha, const_cast<Tensor&>(result), BADDBMM_OP_TYPE);
}

Tensor& addbmm_out_mps(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, Tensor& result) {
  auto b_self = expand_size(self, {batch1.size(1), batch2.size(2)}, "addbmm_out");

  addbmm_or_baddbmm_out_mps_impl(*b_self, batch1, batch2, beta, alpha, result, ADDBMM_OP_TYPE);
  return result;
}

Tensor addbmm_mps(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  Tensor result = at::empty({0}, self.options());
  return addbmm_out_mps(self, batch1, batch2, beta, alpha, result);
}

Tensor &addbmm_mps_(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  return addbmm_out_mps(self, batch1, batch2, beta, alpha, self);
}

} // namespace native
} // namespace at
