//TEST:SIMPLE(filecheck=CHECK_GLSL):  -allow-glsl -stage compute -entry computeMain -target glsl -DTARGET_GLSL
//TEST:SIMPLE(filecheck=CHECK_SPV):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly -DTARGET_SPIRV
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly 
#version 430

// float2 is currently a very new extension; most hardware lacks 
// this extension and will fail the test if attempting to use atomic_float2 
// operations
// #define TEST_when_shader_atomic_float2_is_available

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName
{
    uint data[1];
} outputBuffer;

//TEST_INPUT:ubuffer(data=[0], stride=4):name=int32Buffer
buffer MyBlockName1
{
    int data[1];
} int32Buffer;

//TEST_INPUT:ubuffer(data=[0 0], stride=8):name=int64Buffer
buffer MyBlockName2
{
    uint64_t data[1];
} int64Buffer;

//TEST_INPUT:ubuffer(data=[0], stride=4):name=uint32Buffer
buffer MyBlockName3
{
    uint data[1];
} uint32Buffer;

//TEST_INPUT:ubuffer(data=[0 0], stride=8):name=uint64Buffer
buffer MyBlockName4
{
    uint64_t data[1];
} uint64Buffer;

//TEST_INPUT:ubuffer(data=[0.0], stride=2):name=float16Buffer
buffer MyBlockName5
{
    half data[1];
} float16Buffer;

//TEST_INPUT:ubuffer(data=[0.0], stride=2):name=float32Buffer
buffer MyBlockName6
{
    float data[1];
} float32Buffer;

//TEST_INPUT:ubuffer(data=[0.0 0.0], stride=8):name=float64Buffer
buffer MyBlockName7
{
    double data[1];
} float64Buffer;

// added to tests `out TYPE data` due to Slang bug
bool i32_init(int val, out int data)
{
    data = val;
    return true;
}
bool i32_expect(int val)
{
    return int32Buffer.data[0] == val;
}
bool testAtomicInt32()
{
    return true

        && i32_init(5, int32Buffer.data[0])
        && atomicAdd(int32Buffer.data[0], 1) == 5
        && i32_expect(6)

        && i32_init(5, int32Buffer.data[0])
        && atomicMin(int32Buffer.data[0], 1) == 5
        && i32_expect(1)

        && i32_init(5, int32Buffer.data[0])
        && atomicMax(int32Buffer.data[0], 1) == 5
        && i32_expect(5)

        && i32_init(5, int32Buffer.data[0])
        && atomicExchange(int32Buffer.data[0], 2) == 5
        && i32_expect(2)

        && i32_init(5, int32Buffer.data[0])
        && atomicAnd(int32Buffer.data[0], 1) == 5
        && i32_expect(1)

        && i32_init(5, int32Buffer.data[0])
        && atomicOr(int32Buffer.data[0], 2) == 5
        && i32_expect(7)

        && i32_init(5, int32Buffer.data[0])
        && atomicXor(int32Buffer.data[0], 3) == 5
        && i32_expect(6)

        && i32_init(5, int32Buffer.data[0])
        && atomicCompSwap(int32Buffer.data[0], 5, 2) == 5
        && i32_expect(2)

        && i32_init(5, int32Buffer.data[0])
        && atomicCompSwap(int32Buffer.data[0], 4, 2) == 5
        && i32_expect(5)
        ;
}

bool i64_init(int64_t val, out int64_t data)
{
    data = val;
    return true;
}
bool i64_expect(int64_t val)
{   
    return int64Buffer.data[0] == val;
}
bool testAtomicInt64()
{
    return true

        && i64_init(5, int64Buffer.data[0])
        && atomicAdd(int64Buffer.data[0], 1) == 5
        && i64_expect(6)

        && i64_init(5, int64Buffer.data[0])
        && atomicMin(int64Buffer.data[0], 1) == 5
        && i64_expect(1)

        && i64_init(5, int64Buffer.data[0])
        && atomicMax(int64Buffer.data[0], 1) == 5
        && i64_expect(5)

        && i64_init(5, int64Buffer.data[0])
        && atomicExchange(int64Buffer.data[0], 2) == 5
        && i64_expect(2)

        && i64_init(5, int64Buffer.data[0])
        && atomicAnd(int64Buffer.data[0], 1) == 5
        && i64_expect(1)

        && i64_init(5, int64Buffer.data[0])
        && atomicOr(int64Buffer.data[0], 2) == 5
        && i64_expect(7)

        && i64_init(5, int64Buffer.data[0])
        && atomicXor(int64Buffer.data[0], 3) == 5
        && i64_expect(6)

        && i64_init(5, int64Buffer.data[0])
        && atomicCompSwap(int64Buffer.data[0], 5, 2) == 5
        && i64_expect(2)

        && i64_init(5, int64Buffer.data[0])
        && atomicCompSwap(int64Buffer.data[0], 4, 2) == 5
        && i64_expect(5)
        ;
}

bool u32_init(uint val, out uint data)
{
    data = val;
    return true;
}
bool u32_expect(uint val)
{   
    return uint32Buffer.data[0] == val;
}
bool testAtomicUint32()
{
    return true

        && u32_init(5, uint32Buffer.data[0])
        && atomicAdd(uint32Buffer.data[0], 1) == 5
        && u32_expect(6)

        && u32_init(5, uint32Buffer.data[0])
        && atomicMin(uint32Buffer.data[0], 1) == 5
        && u32_expect(1)

        && u32_init(5, uint32Buffer.data[0])
        && atomicMax(uint32Buffer.data[0], 1) == 5
        && u32_expect(5)

        && u32_init(5, uint32Buffer.data[0])
        && atomicExchange(uint32Buffer.data[0], 2) == 5
        && u32_expect(2)

        && u32_init(5, uint32Buffer.data[0])
        && atomicAnd(uint32Buffer.data[0], 1) == 5
        && u32_expect(1)

        && u32_init(5, uint32Buffer.data[0])
        && atomicOr(uint32Buffer.data[0], 2) == 5
        && u32_expect(7)

        && u32_init(5, uint32Buffer.data[0])
        && atomicXor(uint32Buffer.data[0], 3) == 5
        && u32_expect(6)

        && u32_init(5, uint32Buffer.data[0])
        && atomicCompSwap(uint32Buffer.data[0], 5, 2) == 5
        && u32_expect(2)

        && u32_init(5, uint32Buffer.data[0])
        && atomicCompSwap(uint32Buffer.data[0], 4, 2) == 5
        && u32_expect(5)
        ;
}

bool u64_init(uint64_t val, out uint64_t data)
{
    data = val;
    return true;
}
bool u64_expect(uint64_t val)
{   
    return uint64Buffer.data[0] == val;
}
bool testAtomicUint64()
{
    return true

        && u64_init(5, uint64Buffer.data[0])
        && atomicAdd(uint64Buffer.data[0], 1) == 5
        && u64_expect(6)

        && u64_init(5, uint64Buffer.data[0])
        && atomicMin(uint64Buffer.data[0], 1) == 5
        && u64_expect(1)

        && u64_init(5, uint64Buffer.data[0])
        && atomicMax(uint64Buffer.data[0], 1) == 5
        && u64_expect(5)

        && u64_init(5, uint64Buffer.data[0])
        && atomicExchange(uint64Buffer.data[0], 2) == 5
        && u64_expect(2)

        && u64_init(5, uint64Buffer.data[0])
        && atomicAnd(uint64Buffer.data[0], 1) == 5
        && u64_expect(1)

        && u64_init(5, uint64Buffer.data[0])
        && atomicOr(uint64Buffer.data[0], 2) == 5
        && u64_expect(7)

        && u64_init(5, uint64Buffer.data[0])
        && atomicXor(uint64Buffer.data[0], 3) == 5
        && u64_expect(6)

        && u64_init(5, uint64Buffer.data[0])
        && atomicCompSwap(uint64Buffer.data[0], 5, 2) == 5
        && u64_expect(2)

        && u64_init(5, uint64Buffer.data[0])
        && atomicCompSwap(uint64Buffer.data[0], 4, 2) == 5
        && u64_expect(5)
        ;
}

bool f16_init(half val, out half data)
{
    data = val;
    return true;
}
bool f16_expect(half val)
{   
    return float16Buffer.data[0] == val;
}
bool testAtomicFloat16()
{
    return true

#ifdef TEST_when_shader_atomic_float2_is_available
        && f16_init(5, float16Buffer.data[0])
        && atomicAdd(float16Buffer.data[0], half(1)) == half(5)
        && f16_expect(6)

        && f16_init(5, float16Buffer.data[0])
        && atomicMin(float16Buffer.data[0], half(1)) == half(5)
        && f16_expect(1)

        && f16_init(5, float16Buffer.data[0])
        && atomicMax(float16Buffer.data[0], half(1)) == half(5)
        && f16_expect(5)

        && f16_init(5, float16Buffer.data[0])
        && atomicExchange(float16Buffer.data[0], half(2)) == half(5)
        && f16_expect(2)
#endif // TEST_when_shader_atomic_float2_is_available
        ;
}

bool f32_init(float val, out float data)
{
    data = val;
    return true;
}
bool f32_expect(float val)
{   
    return float32Buffer.data[0] == val;
}
bool testAtomicFloat32()
{
    return true

        && f32_init(5, float32Buffer.data[0])
        && atomicAdd(float32Buffer.data[0], float(1)) == float(5)
        && f32_expect(6)

#ifdef TEST_when_shader_atomic_float2_is_available
        && f32_init(5, float32Buffer.data[0])
        && atomicMin(float32Buffer.data[0], float(1)) == float(5)
        && f32_expect(1)

        && f32_init(5, float32Buffer.data[0])
        && atomicMax(float32Buffer.data[0], float(1)) == float(5)
        && f32_expect(5)

        && f32_init(5, float32Buffer.data[0])
        && atomicExchange(float32Buffer.data[0], float(2)) == float(5)
        && f32_expect(2)
#endif // TEST_when_shader_atomic_float2_is_available
        ;
}

bool f64_init(double val, out double data)
{
    data = val;
    return true;
}
bool f64_expect(double val)
{   
    return float64Buffer.data[0] == val;
}
bool testAtomicFloat64()
{
    return true

        && f64_init(5, float64Buffer.data[0])
        && atomicAdd(float64Buffer.data[0], double(1)) == double(5)
        && f64_expect(6)

#ifdef TEST_when_shader_atomic_float2_is_available
        && f64_init(5, float64Buffer.data[0])
        && atomicMin(float64Buffer.data[0], double(1)) == double(5)
        && f64_expect(1)

        && f64_init(5, float64Buffer.data[0])
        && atomicMax(float64Buffer.data[0], double(1)) == double(5)
        && f64_expect(5)

        && f64_init(5, float64Buffer.data[0])
        && atomicExchange(float64Buffer.data[0], double(2)) == double(5)
        && f64_expect(2)
#endif // TEST_when_shader_atomic_float2_is_available
        ;
}

layout(local_size_x = 1) in;
void computeMain()
{
    // testing has the following pattern in 3 lines per operation:
    // set the value, operation on value, test the result
    outputBuffer.data[0] = true
            && testAtomicInt32()
            && testAtomicInt64()
            && testAtomicUint32()
            && testAtomicUint64()
            && testAtomicFloat16()
            && testAtomicFloat32()
            && testAtomicFloat64()
            ;    
    // CHECK_GLSL: void main(
    // CHECK_SPV: OpEntryPoint
    // BUF: 1
}
