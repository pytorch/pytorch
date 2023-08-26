// Copyright © 2023 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <MetalPerformanceShadersGraph/MPSGraphFourierTransformOps.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

Tensor _fft_r2c_mps(const Tensor& self, int64_t signal_ndim, bool normalized, bool onesided) {
    using namespace mps;
    
    // Ensure the input tensor is on the expected device and has the expected datatype
    TORCH_CHECK(self.is_mps(), "_fft_r2c_mps: Expected MPS tensor");
    
    // Define the CachedGraph for the FFT operation
    struct CachedGraph : public MPSCachedGraph {
        CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor* inputTensor_ = nil;
        MPSGraphTensor* outputTensor_ = nil;
    };
    
    // Get the current MPS stream
    MPSStream* stream = getCurrentMPSStream();
    
    @autoreleasepool {
        // Define a unique key for the FFT operation
        string key = "fft_r2c_" + getTensorsStringKey({self}) + getMPSTypeString(self);
        
        // Look up or create the cached graph for the FFT operation
        auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
            MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));
            newCachedGraph->inputTensor_ = inputTensor;
            
            // Define the FFT operation on the MPS graph
            MPSGraphTensor* outputTensor = [mpsGraph FFTWithSourceTensor:inputTensor
                                                               dimension:signal_ndim
                                                              normalized:normalized
                                                                onesided:onesided
                                                                    name:nil];
            newCachedGraph->outputTensor_ = outputTensor;
        });
        
        // Set up the input tensor placeholder
        Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
        
        // Define the feeds dictionary
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[[NSMutableDictionary alloc] init] autorelease];
        [feeds setObject:inputPlaceholder.getMPSGraphTensorData() forKey:inputPlaceholder.getMPSGraphTensor()];
        
        // Define the results dictionary
        Tensor output_out = at::empty_like(self); // Placeholder for the output tensor
        Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_out);
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [[[NSMutableDictionary alloc] init] autorelease];
        [results setObject:outputPlaceholder.getMPSGraphTensorData() forKey:outputPlaceholder.getMPSGraphTensor()];
        
        // Execute the MPS graph
        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
        
        return output_out;
    }
}

} // namespace native
} // namespace at


Tensor _fft_r2c_mps_out(const Tensor& input, IntArrayRef dim, int normalization, bool onesided, Tensor& out) {
    // TODO: Implement the FFT R2C out operation for MPS backend
    TORCH_CHECK(false, "_fft_r2c_mps_out is not yet implemented for MPS backend");
    return out;
}

Tensor _fft_c2r_mps(const Tensor& input, IntArrayRef dim, int normalization, int64_t last_dim_size) {
    // TODO: Implement the FFT C2R operation for MPS backend
    TORCH_CHECK(false, "_fft_c2r_mps is not yet implemented for MPS backend");
    return input;
}

Tensor _fft_c2r_mps_out(const Tensor& input, IntArrayRef dim, int normalization, int64_t last_dim_size, Tensor& out) {
    // TODO: Implement the FFT C2R out operation for MPS backend
    TORCH_CHECK(false, "_fft_c2r_mps_out is not yet implemented for MPS backend");
    return out;
}

Tensor _fft_c2c_mps(const Tensor& input, IntArrayRef dim, int normalization, bool forward) {
    // TODO: Implement the FFT C2C operation for MPS backend
    TORCH_CHECK(false, "_fft_c2c_mps is not yet implemented for MPS backend");
    return input;
}

Tensor _fft_c2c_mps_out(const Tensor& input, IntArrayRef dim, int normalization, bool forward, Tensor& out) {
    // TODO: Implement the FFT C2C out operation for MPS backend
    TORCH_CHECK(false, "_fft_c2c_mps_out is not yet implemented for MPS backend");
    return out;
}
