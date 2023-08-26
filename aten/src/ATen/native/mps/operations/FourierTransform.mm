// Copyright © 2023 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <MetalPerformanceShadersGraph/MPSGraphFourierTransformOps.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

namespace at::native {

// Real-to-Hermitean FFT
Tensor _fft_r2c_mps(const Tensor& self, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
    using namespace mps;
    
    TORCH_CHECK(self.is_mps(), "_fft_r2c_mps: Expected MPS tensor");
    
    struct CachedGraph : public MPSCachedGraph {
        CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor* inputTensor_ = nil;
        MPSGraphTensor* outputTensor_ = nil;
    };
    
    MPSStream* stream = getCurrentMPSStream();
    
    @autoreleasepool {
        string key = "fft_r2c_" + getTensorsStringKey({self}) + getMPSTypeString(self);
        
        auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
            MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));
            newCachedGraph->inputTensor_ = inputTensor;
            
            // Create FFT descriptor
            MPSGraphFFTDescriptor* descriptor = [MPSGraphFFTDescriptor descriptor];
            descriptor.inverse = NO;
            descriptor.scalingMode = normalized ? MPSGraphFFTScalingModeUnitary : MPSGraphFFTScalingModeNone;
            
            // Use the realToHermiteanFFTWithTensor method from MPSGraphFourierTransformOps.h
            MPSGraphTensor* outputTensor = [mpsGraph realToHermiteanFFTWithTensor:inputTensor
                                                                             axes:@[@(signal_ndim)]
                                                                       descriptor:descriptor
                                                                             name:nil];
            newCachedGraph->outputTensor_ = outputTensor;
        });
        
        Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
        
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[[NSMutableDictionary alloc] init] autorelease];
        [feeds setObject:inputPlaceholder.getMPSGraphTensorData() forKey:inputPlaceholder.getMPSGraphTensor()];
        
        Tensor output = at::empty(signal_sizes, self.options());
        Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [[[NSMutableDictionary alloc] init] autorelease];
        [results setObject:outputPlaceholder.getMPSGraphTensorData() forKey:outputPlaceholder.getMPSGraphTensor()];
        
        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
        
        return output;
    }
}

// Real-to-Hermitean FFT, writing result to the provided output tensor
Tensor& _fft_r2c_mps_out(const Tensor& self, Tensor& out, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
    using namespace mps;
    
    TORCH_CHECK(self.is_mps(), "_fft_r2c_mps_out: Expected MPS tensor");
    TORCH_CHECK(out.is_mps(), "_fft_r2c_mps_out: Expected MPS tensor for output");
    
    struct CachedGraph : public MPSCachedGraph {
        CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor* inputTensor_ = nil;
        MPSGraphTensor* outputTensor_ = nil;
    };
    
    MPSStream* stream = getCurrentMPSStream();
    
    @autoreleasepool {
        string key = "fft_r2c_out_" + getTensorsStringKey({self, out}) + getMPSTypeString(self);
        
        auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
            MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));
            newCachedGraph->inputTensor_ = inputTensor;
            
            // Create FFT descriptor
            MPSGraphFFTDescriptor* descriptor = [MPSGraphFFTDescriptor descriptor];
            descriptor.inverse = NO;
            descriptor.scalingMode = normalized ? MPSGraphFFTScalingModeUnitary : MPSGraphFFTScalingModeNone;
            
            // Use the realToHermiteanFFTWithTensor method from MPSGraphFourierTransformOps.h
            MPSGraphTensor* outputTensor = [mpsGraph realToHermiteanFFTWithTensor:inputTensor
                                                                             axes:@[@(signal_ndim)]
                                                                       descriptor:descriptor
                                                                             name:nil];
            newCachedGraph->outputTensor_ = outputTensor;
        });
        
        Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
        
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[[NSMutableDictionary alloc] init] autorelease];
        [feeds setObject:inputPlaceholder.getMPSGraphTensorData() forKey:inputPlaceholder.getMPSGraphTensor()];
        
        Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [[[NSMutableDictionary alloc] init] autorelease];
        [results setObject:outputPlaceholder.getMPSGraphTensorData() forKey:outputPlaceholder.getMPSGraphTensor()];
        
        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
        
        return out;
    }
}

// Hermitean-to-Real FFT
Tensor _fft_c2r_mps(const Tensor& self, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
    using namespace mps;
    
    TORCH_CHECK(self.is_mps(), "_fft_c2r_mps: Expected MPS tensor");
    
    struct CachedGraph : public MPSCachedGraph {
        CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor* inputTensor_ = nil;
        MPSGraphTensor* outputTensor_ = nil;
    };
    
    MPSStream* stream = getCurrentMPSStream();
    
    @autoreleasepool {
        string key = "fft_c2r_" + getTensorsStringKey({self}) + getMPSTypeString(self);
        
        auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
            MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));
            newCachedGraph->inputTensor_ = inputTensor;
            
            // Create FFT descriptor
            MPSGraphFFTDescriptor* descriptor = [MPSGraphFFTDescriptor descriptor];
            descriptor.inverse = YES;
            descriptor.scalingMode = normalized ? MPSGraphFFTScalingModeUnitary : MPSGraphFFTScalingModeNone;
            
            // Use the HermiteanToRealFFTWithTensor method from MPSGraphFourierTransformOps.h
            MPSGraphTensor* outputTensor = [mpsGraph HermiteanToRealFFTWithTensor:inputTensor
                                                                             axes:@[@(signal_ndim)]
                                                                       descriptor:descriptor
                                                                             name:nil];
            newCachedGraph->outputTensor_ = outputTensor;
        });
        
        Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
        
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[[NSMutableDictionary alloc] init] autorelease];
        [feeds setObject:inputPlaceholder.getMPSGraphTensorData() forKey:inputPlaceholder.getMPSGraphTensor()];
        
        Tensor output = at::empty(signal_sizes, self.options());
        Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [[[NSMutableDictionary alloc] init] autorelease];
        [results setObject:outputPlaceholder.getMPSGraphTensorData() forKey:outputPlaceholder.getMPSGraphTensor()];
        
        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
        
        return output;
    }
}

// Hermitean-to-Real FFT, writing result to the provided output tensor
Tensor& _fft_c2r_mps_out(const Tensor& self, Tensor& out, int64_t signal_ndim, bool normalized, IntArrayRef signal_sizes) {
    using namespace mps;
    
    TORCH_CHECK(self.is_mps(), "_fft_c2r_mps_out: Expected MPS tensor");
    TORCH_CHECK(out.is_mps(), "_fft_c2r_mps_out: Expected MPS tensor for output");
    
    struct CachedGraph : public MPSCachedGraph {
        CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor* inputTensor_ = nil;
        MPSGraphTensor* outputTensor_ = nil;
    };
    
    MPSStream* stream = getCurrentMPSStream();
    
    @autoreleasepool {
        string key = "fft_c2r_out_" + getTensorsStringKey({self, out}) + getMPSTypeString(self);
        
        auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
            MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));
            newCachedGraph->inputTensor_ = inputTensor;
            
            // Create FFT descriptor
            MPSGraphFFTDescriptor* descriptor = [MPSGraphFFTDescriptor descriptor];
            descriptor.inverse = YES;
            descriptor.scalingMode = normalized ? MPSGraphFFTScalingModeUnitary : MPSGraphFFTScalingModeNone;
            
            // Use the HermiteanToRealFFTWithTensor method from MPSGraphFourierTransformOps.h
            MPSGraphTensor* outputTensor = [mpsGraph HermiteanToRealFFTWithTensor:inputTensor
                                                                             axes:@[@(signal_ndim)]
                                                                       descriptor:descriptor
                                                                             name:nil];
            newCachedGraph->outputTensor_ = outputTensor;
        });
        
        Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
        
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[[NSMutableDictionary alloc] init] autorelease];
        [feeds setObject:inputPlaceholder.getMPSGraphTensorData() forKey:inputPlaceholder.getMPSGraphTensor()];
        
        Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [[[NSMutableDictionary alloc] init] autorelease];
        [results setObject:outputPlaceholder.getMPSGraphTensorData() forKey:outputPlaceholder.getMPSGraphTensor()];
        
        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
        
        return out;
    }
}

// complex-to-complex FFT
Tensor _fft_c2c_mps(const Tensor& self, int64_t signal_ndim, bool normalized, bool forward, IntArrayRef signal_sizes) {
    using namespace mps;
    
    TORCH_CHECK(self.is_mps(), "_fft_c2c_mps: Expected MPS tensor");
    
    struct CachedGraph : public MPSCachedGraph {
        CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor* inputTensor_ = nil;
        MPSGraphTensor* outputTensor_ = nil;
    };
    
    MPSStream* stream = getCurrentMPSStream();
    
    @autoreleasepool {
        string key = "fft_c2c_" + getTensorsStringKey({self}) + getMPSTypeString(self);
        
        auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
            MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));
            newCachedGraph->inputTensor_ = inputTensor;
            
            // Create FFT descriptor
            MPSGraphFFTDescriptor* descriptor = [MPSGraphFFTDescriptor descriptor];
            descriptor.inverse = !forward; // Set inverse based on the 'forward' flag
            descriptor.scalingMode = normalized ? MPSGraphFFTScalingModeUnitary : MPSGraphFFTScalingModeNone;
            
            // Use the fastFourierTransformWithTensor method from MPSGraphFourierTransformOps.h
            MPSGraphTensor* outputTensor = [mpsGraph fastFourierTransformWithTensor:inputTensor
                                                                               axes:@[@(signal_ndim)]
                                                                         descriptor:descriptor
                                                                               name:nil];
            newCachedGraph->outputTensor_ = outputTensor;
        });
        
        Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
        
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[[NSMutableDictionary alloc] init] autorelease];
        [feeds setObject:inputPlaceholder.getMPSGraphTensorData() forKey:inputPlaceholder.getMPSGraphTensor()];
        
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [[[NSMutableDictionary alloc] init] autorelease];
        
        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
        
        MPSGraphTensorData* outputData = [results objectForKey:cachedGraph->outputTensor_];
        Tensor output = createTensorFromMPSGraphTensorData(outputData);
        
        return output;
    }
}

// complex-to-complex FFT, writng result to the provided output tensor
Tensor& _fft_c2c_mps_out(const Tensor& self, Tensor& out, int64_t signal_ndim, bool normalized, bool forward, IntArrayRef signal_sizes) {
    using namespace mps;
    
    TORCH_CHECK(self.is_mps() && out.is_mps(), "_fft_c2c_mps_out: Expected MPS tensors for both input and output");
    
    struct CachedGraph : public MPSCachedGraph {
        CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor* inputTensor_ = nil;
        MPSGraphTensor* outputTensor_ = nil;
    };
    
    MPSStream* stream = getCurrentMPSStream();
    
    @autoreleasepool {
        string key = "fft_c2c_out_" + getTensorsStringKey({self, out}) + getMPSTypeString(self);
        
        auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
            MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));
            newCachedGraph->inputTensor_ = inputTensor;
            
            // Create FFT descriptor
            MPSGraphFFTDescriptor* descriptor = [MPSGraphFFTDescriptor descriptor];
            descriptor.inverse = !forward; // Set inverse based on the 'forward' flag
            descriptor.scalingMode = normalized ? MPSGraphFFTScalingModeUnitary : MPSGraphFFTScalingModeNone;
            
            // Use the fastFourierTransformWithTensor method from MPSGraphFourierTransformOps.h
            MPSGraphTensor* outputTensor = [mpsGraph fastFourierTransformWithTensor:inputTensor
                                                                               axes:@[@(signal_ndim)]
                                                                         descriptor:descriptor
                                                                               name:nil];
            newCachedGraph->outputTensor_ = outputTensor;
        });
        
        Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
        
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[[NSMutableDictionary alloc] init] autorelease];
        [feeds setObject:inputPlaceholder.getMPSGraphTensorData() forKey:inputPlaceholder.getMPSGraphTensor()];
        
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [[[NSMutableDictionary alloc] init] autorelease];
        
        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
        
        MPSGraphTensorData* outputData = [results objectForKey:cachedGraph->outputTensor_];
        Tensor output = createTensorFromMPSGraphTensorData(outputData);
        
        // Copy the result to the provided output tensor
        out.copy_(output);
    }
    
    return out;
}

} // namespace at::native