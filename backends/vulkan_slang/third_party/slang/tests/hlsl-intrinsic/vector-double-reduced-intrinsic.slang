//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
// TODO(JS):
// On CI systems DX11 test failed, so disable for now
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -output-using-type -shaderobj -render-feature hardware-device
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -shaderobj -render-feature double
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<double4> outputBuffer;

typedef double Float;

typedef vector<Float, 3> FloatVector;
typedef vector<int, 3> IntVector;
typedef vector<uint, 3> UIntVector;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);

    Float vf = idx * (1.0f / (4.0f));
     
    FloatVector f = FloatVector(0.1f, vf, vf + 0.2f);
    
    // Operate over all values
    FloatVector ft = {};

    // TODO(JS):
    // fmod - disabled not available on D3D for double
    //ft += FloatVector(IntVector(((f % 0.11f) * 100) + 0.5));
            
    {
        // Vector specific
        FloatVector nv = normalize(f);
        ft += nv;
        
        FloatVector perp = cross(nv, f.zxy); 
        ft += perp;
  
           
        // TODO(JS): 
        // Disabled because dotwith a double vector fails on dxc, with validation error (!)
        //ft += dot(perp, f.zyx);

        ft += length(f);
  
        // TODO(JS):
        // Disabled because reflect with double fails on dxc
        //ft += reflect(f, FloatVector(0, 1, 0));
    }    
    
    // Reduced set of intrinsics that should work on all targets
    
#if 0   
    // TODO(JS):
    // Disabled because not supported on VK (glsl) in vector form    
    ft += rcp(1.0l + f);
#endif
    
    ft += sign(f - 0.5l);
    
    ft += saturate(f * 4 - 2.0l);
       
    ft += abs(f * 4 - 2.0l);
           
    ft += min(0.5l, f);
    ft += max(f, 0.75l);

    ft += clamp(f, 0.1l, 0.3l);
        
    outputBuffer[idx] = vector<Float, 4>(ft, 0);
}