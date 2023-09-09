#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <MetalPerformanceShadersGraph/MPSGraphFourierTransformOps.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else

#endif

namespace at::native {

enum class FFTType {
    R2C,
    C2R,
    C2C
};

// Prototypes


Tensor _fft_r2c_mps(const Tensor& self, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes);
Tensor& _fft_r2c_mps_out(const Tensor& self, Tensor& out, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes);
Tensor _fft_c2r_mps(const Tensor& self, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes);
Tensor& _fft_c2r_mps_out(const Tensor& self, Tensor& out, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes);
Tensor _fft_c2c_mps(const Tensor& self, int64_t signal_ndim, bool normalized, bool forward, IntArrayRef signal_sizes);
Tensor& _fft_c2c_mps_out(const Tensor& self, Tensor& out, int64_t signal_ndim, bool normalized, bool forward, IntArrayRef signal_sizes);
static Tensor runFFTGraph(const Tensor& self, const Tensor& out, bool out_provided, const std::string& key_prefix, FFTType fftType, int64_t signal_ndim, bool normalized, bool forward);


using namespace mps;


static MPSGraphTensor* createFFTGraph(MPSGraph* graph, MPSGraphTensor* inputTensor, int64_t signal_ndim, bool normalized, FFTType fftType, bool forward) {
    
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

static Tensor runFFTGraph(const Tensor& self, const Tensor& out, bool out_provided, const std::string& key_prefix, FFTType fftType, int64_t signal_ndim, bool normalized, bool forward) {
    
    TORCH_CHECK(self.is_mps(), key_prefix + ": Expected MPS tensor");
    if (out_provided) {
        TORCH_CHECK(out.is_mps(), key_prefix + ": Expected MPS tensor for output");
    }
    
    MPSStream* stream = getCurrentMPSStream();

    struct CachedGraph : public MPSCachedGraph {
        CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor* inputTensor_ = nil;
        MPSGraphTensor* outputTensor_ = nil;
    };
    
    @autoreleasepool {

        string key = key_prefix + getTensorsStringKey({self}) + getMPSTypeString(self);
        
        auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {

            MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));
            MPSGraphTensor* outputTensor = createFFTGraph(mpsGraph, inputTensor, signal_ndim, normalized, fftType, forward);
            newCachedGraph->inputTensor_ = inputTensor;
            newCachedGraph->outputTensor_ = outputTensor;
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
    return runFFTGraph(self, {}, false, "fft_r2c_mps", FFTType::R2C, signal_ndim, normalized, true);
}

// Real-to-Hermitean FFT, writing result to the provided output tensor
Tensor& _fft_r2c_mps_out(const Tensor& self, Tensor& out, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
    runFFTGraph(self, out, true, "fft_r2c_out_mps", FFTType::R2C, signal_ndim, normalized, true);
    return out;
}

// Hermitean-to-Real FFT
Tensor _fft_c2r_mps(const Tensor& self, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
    return runFFTGraph(self, {}, false, "fft_c2r_mps", FFTType::C2R, signal_ndim, normalized, false);
}

// Hermitean-to-Real FFT, writing result to the provided output tensor
Tensor& _fft_c2r_mps_out(const Tensor& self, Tensor& out, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
    runFFTGraph(self, out, true, "fft_c2r_out_mps", FFTType::C2R, signal_ndim, normalized, false);
    return out;
}

// complex-to-complex FFT
Tensor _fft_c2c_mps(const Tensor& self, int64_t signal_ndim, bool normalized, bool forward, IntArrayRef signal_sizes) {
    return runFFTGraph(self, {}, false, "fft_c2c_mps", FFTType::C2C, signal_ndim, normalized, forward);
}

// complex-to-complex FFT, writing result to the provided output tensor
Tensor& _fft_c2c_mps_out(const Tensor& self, Tensor& out, int64_t signal_ndim, bool normalized, bool forward, IntArrayRef signal_sizes) {
    runFFTGraph(self, out, true, "fft_c2c_out_mps", FFTType::C2C, signal_ndim, normalized, forward);
    return out;
}

} // namespace at::native
