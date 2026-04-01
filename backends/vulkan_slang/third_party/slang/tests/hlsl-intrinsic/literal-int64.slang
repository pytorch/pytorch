//TEST(compute):COMPARE_COMPUTE_EX(filecheck=CHK):-cpu -compute -shaderobj
// No support for int64_t on D3D11 (no sm 6.0)
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX(filecheck=CHK):-slang -compute -shaderobj
// No support with Dx12 with dxbc. Needs SM6.0 + dxil
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX(filecheck=CHK):-slang -compute -dx12 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck=CHK):-slang -compute -profile cs_6_0 -dx12 -use-dxil -shaderobj -render-feature hardware-device
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck=CHK):-vk -compute -shaderobj -render-feature int64
//TEST(compute):COMPARE_COMPUTE_EX(filecheck=CHK):-cuda -compute -shaderobj

// The expected behavior is that the literal type is the first from the following list
// in which the value can fit:
// - For decimal bases:
//     - `int`
//     - `int64_t`
// - For non-decimal bases:
//     - `int`
//     - `uint`
//     - `int64_t`
//     - `uint64_t`

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint64_t> outputBuffer;

[numthreads(10, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(int64_t(dispatchThreadID.x));
    
    uint64_t v = 0;
    
    if (idx == 0)
    {
        // Should be 0xffffffffffffffff
        v = -1;
    }
    else if (idx == 1)
    {
        // Should be 0x00000000ffffffff
        v = -1u;
    }
    else if (idx == 2)
    {
        // Should be 0x0000000080000000
        v = 0x80000000u;
    }
    else if (idx == 3)
    {
        // Should be 0x0000000080000000
        int64_t v0 = 0x80000000;
        v = v0;
    }
    else if (idx == 4)
    {
        // Should be 0x0000000080000000. Shows hexadecimal literals can represent unsigned 32-bit integers.
        v = 0x80000000;
    }
    else if (idx == 5)
    {
        // Should be 0xffffffffffffffff
        v = ~0; 
    }
    else if (idx == 6)
    {
        // Should be 0xffffffffffffffff
        v = ~0LL; 
    }
    else if (idx == 7)
    {
        // Should be 0x7fffffffffffffff. Shows decimal literals can represent 64-bit integers.
        v = 9223372036854775807;
    }
    else if (idx == 8)
    {
        // Should be 1. Shows decimal literals can represent INT64_MIN or 0x8000000000000000.
        // Warning will be emitted as negative(-) is scanned separately in the lexer, and the positive
        // literal portion will emit a warning.
        // The final value must be correctly set as INT64_MIN and must be negative.
        if (-9223372036854775808 < 0L)
        {
            v = 1;
        }
        else
        {
            v = 0;
        }
        //CHK: warning 39999: integer literal is too large to be represented in a signed integer type, interpreting as unsigned
    }
    else
    {
        // Shoud be 0xfaaaaaaabaaaaaaa. Shows hexadecimal literals can represent unsigned 64-bit integers.
        v = 0xfaaaaaaabaaaaaaa;
    }
    
    outputBuffer[idx] = v;
}
