#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <MetalPerformanceShadersGraph/MPSGraphFourierTransformOps.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

namespace at::native {

enum class FFTType {
    R2C,
    C2R,
    C2C
};

MPSGraphTensor* createFFTGraph(MPSGraph* graph, const Tensor& tensor, int64_t signal_ndim, bool normalized, FFTType fftType, bool forward) {
    MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(graph, getMPSDataType(tensor), getMPSShape(tensor));
    
    MPSGraphFFTDescriptor* descriptor = [MPSGraphFFTDescriptor descriptor];
    descriptor.inverse = (fftType == FFTType::C2C) ? !forward : (fftType == FFTType::C2R);
    descriptor.scalingMode = normalized ? MPSGraphFFTScalingModeUnitary : MPSGraphFFTScalingModeNone;
    
    switch (fftType) {
        case FFTType::R2C:
            return [graph realToHermiteanFFTWithTensor:inputTensor axes:@[@(signal_ndim)] descriptor:descriptor name:nil];
        case FFTType::C2R:
            return [graph HermiteanToRealFFTWithTensor:inputTensor axes:@[@(signal_ndim)] descriptor:descriptor name:nil];
        case FFTType::C2C:
            return [graph fastFourierTransformWithTensor:inputTensor axes:@[@(signal_ndim)] descriptor:descriptor name:nil];
    }
}

Tensor runFFTGraph(const Tensor& self, const Tensor& out, bool out_provided, const std::string& key_prefix, FFTType fftType, int64_t signal_ndim, bool normalized, bool forward) {
    using namespace mps;
    
    TORCH_CHECK(self.is_mps(), key_prefix + ": Expected MPS tensor");
    if (out_provided) {
        TORCH_CHECK(out.is_mps(), key_prefix + ": Expected MPS tensor for output");
    }
    
    MPSStream* stream = getCurrentMPSStream();
    
    @autoreleasepool {
        string key = key_prefix + getTensorsStringKey({self}) + getMPSTypeString(self);
        
        auto cachedGraph = LookUpOrCreateCachedGraph<MPSCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
            newCachedGraph->outputTensor_ = createFFTGraph(mpsGraph, self, signal_ndim, normalized, fftType, forward);
        });
        
        Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
        
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[[NSMutableDictionary alloc] init] autorelease];
        [feeds setObject:inputPlaceholder.getMPSGraphTensorData() forKey:inputPlaceholder.getMPSGraphTensor()];
        
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [[[NSMutableDictionary alloc] init] autorelease];
        
        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
        
        MPSGraphTensorData* outputData = [results objectForKey:cachedGraph->outputTensor_];
        Tensor output = out_provided ? out : createTensorFromMPSGraphTensorData(outputData);
        
        if (out_provided) {
            out.copy_(output);
        }
        
        return output;
    }
}

// Real-to-Hermitean FFT
Tensor _fft_r2c_mps(const Tensor& self, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
    return runFFTGraph(self, {}, false, "r2c", FFTType::R2C, signal_ndim, normalized, true);
}

// Real-to-Hermitean FFT, writing result to the provided output tensor
Tensor& _fft_r2c_mps_out(const Tensor& self, Tensor& out, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
    runFFTGraph(self, out, true, "r2c_out", FFTType::R2C, signal_ndim, normalized, true);
    return out;
}

// Hermitean-to-Real FFT
Tensor _fft_c2r_mps(const Tensor& self, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
    return runFFTGraph(self, {}, false, "c2r", FFTType::C2R, signal_ndim, normalized, false);
}

// Hermitean-to-Real FFT, writing result to the provided output tensor
Tensor& _fft_c2r_mps_out(const Tensor& self, Tensor& out, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
    runFFTGraph(self, out, true, "c2r_out", FFTType::C2R, signal_ndim, normalized, false);
    return out;
}

// complex-to-complex FFT
Tensor _fft_c2c_mps(const Tensor& self, int64_t signal_ndim, bool normalized, bool forward, IntArrayRef signal_sizes) {
    return runFFTGraph(self, {}, false, "c2c", FFTType::C2C, signal_ndim, normalized, forward);
}

// complex-to-complex FFT, writing result to the provided output tensor
Tensor& _fft_c2c_mps_out(const Tensor& self, Tensor& out, int64_t signal_ndim, bool normalized, bool forward, IntArrayRef signal_sizes) {
    runFFTGraph(self, out, true, "c2c_out", FFTType::C2C, signal_ndim, normalized, forward);
    return out;
}

} // namespace at::native
