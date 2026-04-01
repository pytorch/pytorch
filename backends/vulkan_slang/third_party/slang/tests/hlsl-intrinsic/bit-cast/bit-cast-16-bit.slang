// bit-cast-16-bit.slang

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile sm_6_2 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -render-feature int16,half
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16], stride=4):name inputBuffer
RWStructuredBuffer<int> inputBuffer;

int16_t readI(inout int index) { return int16_t(inputBuffer[(index++) & 0xF]); }
uint16_t readU(inout int index) { return int16_t(inputBuffer[(index++) & 0xF]); }
float16_t readF(inout int index) { return float16_t(float(inputBuffer[(index++) & 0xF])); }

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

void write(int initial, inout int index, uint value)
{
    let tmp = index++;
    if((tmp & 3) == initial)
    {
        outputBuffer[tmp & 0xF] = int(value);
    }
}

void write(int initial, inout int index, uint16_t value)
{
    write(initial, index, uint(value));
}

void write(int initial, inout int index, int16_t value)
{
    write(initial, index, uint(int(value)));
}

void write(int initial, inout int index, float16_t value)
{
    write(initial, index, asuint(float(value)));
}


void test(int initial)
{
    int input = initial;
    int output = 0;

    // Scalar
    {
        let i = readI(input);
        let u = readU(input);
        let f = readF(input);

        // int->float
        let a = asfloat16(i);
        write(initial, output, a);

        // float->uint
        let b = asuint16(f);
        write(initial, output, b);

        // uint->int
        let c = asint16(u);
        write(initial, output, c);

        // float->float
        let d = asfloat16(f);
        write(initial, output, d);
    }

    // Vector
    {
        let i = int16_t2(readI(input), readI(input));
        let u = uint16_t2(readU(input), readU(input));
        let f = float16_t2(readF(input), readF(input));

        // uint->float
        let a = asfloat16(u);
        write(initial, output, a.x);
        write(initial, output, a.y);

        // float->int
        let b = asint16(f);
        write(initial, output, b.x);
        write(initial, output, b.y);

        // int->uint
        let c = asuint16(i);
        write(initial, output, c.x);
        write(initial, output, c.y);

        // int->int
        let d = asint16(i);
        write(initial, output, d.x);
        write(initial, output, d.y);
    }

}


[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    test(dispatchThreadID.x);
}
