//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -output-using-type -dx12 -profile cs_6_6 -use-dxil
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -shaderobj -output-using-type -emit-spirv-directly
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cuda -compute -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-slang -shaderobj -mtl -output-using-type

// Slang removes parentheses characters for the bitwise operators when they are not needed.
// DXC prints warning messages even when the expression is correct.
// We are snoozing the warning and testing it here to prevent a regression.

//TEST_INPUT: ubuffer(data=[0 1 2 3 4 5 6 7 8 9], stride=4):name inputBuffer
RWStructuredBuffer<int> inputBuffer;

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// -----------+----------+------------------------------+---------------
// Precedence | Operator | Description                  | Associativity
// -----------+----------+------------------------------+---------------
// 4          | + -      | Addition and subtraction     | Left-to-right
// -----------+----------+------------------------------+
// 5          | << >>    | Bitwise left and right shift |
// -----------+----------+------------------------------+
// 8          | &        | Bitwise AND                  |
// -----------+----------+------------------------------+
// 9          | ^        | Bitwise XOR (exclusive or)   |
// -----------+----------+------------------------------+
// 10         | |        | Bitwise OR (inclusive or)    |
// -----------+----------+------------------------------+---------------

bool Test_And_Or()
{
    uint32_t a = inputBuffer[1];
    uint32_t b = inputBuffer[3];
    uint32_t c = inputBuffer[2];
    uint32_t d = inputBuffer[6];

    return true
        && 3 == ((a & b) | (c & d))
        && 0 == (a & (b | c) & d)
        && 2 == (((a & b) | c) & d)
        && 1 == (a & (b | (c & d)))
        && 3 == (a & b | c & d)

        && 2 == ((a | b) & (c | d))
        && 7 == (a | (b & c) | d)
        && 6 == (((a | b) & c) | d)
        && 3 == (a | (b & (c | d)))
        && 7 == (a | b & c | d)
        ;
}

bool Test_And_Xor()
{
    uint32_t a = inputBuffer[1];
    uint32_t b = inputBuffer[3];
    uint32_t c = inputBuffer[2];
    uint32_t d = inputBuffer[6];

    return true
        && 3 == ((a & b) ^ (c & d))
        && 0 == (a & (b ^ c) & d)
        && 2 == (((a & b) ^ c) & d)
        && 1 == (a & (b ^ (c & d)))
        && 3 == (a & b ^ c & d)

        && 0 == ((a ^ b) & (c ^ d))
        && 5 == (a ^ (b & c) ^ d)
        && 4 == (((a ^ b) & c) ^ d)
        && 1 == (a ^ (b & (c ^ d)))
        && 5 == (a ^ b & c ^ d)
        ;
}

bool Test_Xor_Or()
{
    uint32_t a = inputBuffer[1];
    uint32_t b = inputBuffer[4];
    uint32_t c = inputBuffer[2];
    uint32_t d = inputBuffer[7];

    return true
        && 5 == ((a ^ b) | (c ^ d))
        && 0 == (a ^ (b | c) ^ d)
        && 0 == (((a ^ b) | c) ^ d)
        && 4 == (a ^ (b | (c ^ d)))
        && 5 == (a ^ b | c ^ d)

        && 2 == ((a | b) ^ (c | d))
        && 7 == (a | (b ^ c) | d)
        && 7 == (((a | b) ^ c) | d)
        && 3 == (a | (b ^ (c | d)))
        && 7 == (a | b ^ c | d)
        ;
}

bool Test_LShift_RShift()
{
    uint32_t a = inputBuffer[4];
    uint32_t b = inputBuffer[1];
    uint32_t c = inputBuffer[2];
    uint32_t d = inputBuffer[3];

    return true
        && 0 == ((a << b) >> (c << d))
        && 32 == (a << (b >> c) << d)
        && 16 == (((a << b) >> c) << d)
        && 4 == (a << (b >> (c << d)))
        && 16 == (a << b >> c << d)

        && 2 == ((a >> b) << (c >> d))
        && 0 == (a >> (b << c) >> d)
        && 1 == (((a >> b) << c) >> d)
        && 2 == (a >> (b << (c >> d)))
        && 1 == (a >> b << c >> d)
        ;
}

bool Test_LShift_And()
{
    uint32_t a = inputBuffer[1];
    uint32_t b = inputBuffer[5];
    uint32_t c = inputBuffer[2];
    uint32_t d = inputBuffer[4];

    return true
        && 32 == ((a << b) & (c << d))
        && 16 == (a << (b & c) << d)
        && 0 == (((a << b) & c) << d)
        && 1 == (a << (b & (c << d)))
        && 32 == (a << b & c << d)

        && 1 == ((a & b) << (c & d))
        && 0 == (a & (b << c) & d)
        && 4 == (((a & b) << c) & d)
        && 1 == (a & (b << (c & d)))
        && 0 == (a & b << c & d)
        ;
}

bool Test_LShift_Or()
{
    uint32_t a = inputBuffer[1];
    uint32_t b = inputBuffer[2];
    uint32_t c = inputBuffer[4];
    uint32_t d = inputBuffer[3];

    return true
        && 36 == ((a << b) | (c << d))
        && 512 == (a << (b | c) << d)
        && 32 == (((a << b) | c) << d)
        && 1073741824 == (a << ((b | (c << d)) - 4))
        && 36 == (a << b | c << d)

        && 384 == ((a | b) << (c | d))
        && 35 == (a | (b << c) | d)
        && 51 == (((a | b) << c) | d)
        && 257 == (a | (b << (c | d)))
        && 35 == (a | b << c | d)
        ;
}

bool Test_LShift_Xor()
{
    uint32_t a = inputBuffer[1];
    uint32_t b = inputBuffer[2];
    uint32_t c = inputBuffer[4];
    uint32_t d = inputBuffer[3];

    return true
        && 36 == ((a << b) ^ (c << d))
        && 512 == (a << (b ^ c) << d)
        && 0 == (((a << b) ^ c) << d)
        && 1073741824 == ((a << ((b ^ (c << d)) - 4)))
        && 36 == (a << b ^ c << d)

        && 384 == ((a ^ b) << (c ^ d))
        && 34 == (a ^ (b << c) ^ d)
        && 51 == (((a ^ b) << c) ^ d)
        && 257 == (a ^ (b << (c ^ d)))
        && 34 == (a ^ b << c ^ d)
        ;
}

bool Test_Add_LShift()
{
    uint32_t a = inputBuffer[1];
    uint32_t b = inputBuffer[3];
    uint32_t c = inputBuffer[2];
    uint32_t d = inputBuffer[4];

    return true
        && 256 == ((a + b) << (c + d))
        && 17 == (a + (b << c) + d)
        && 20 == (((a + b) << c) + d)
        && 193 == (a + (b << (c + d)))
        && 256 == (a + b << c + d)

        && 40 == ((a << b) + (c << d))
        && 512 == (a << (b + c) << d)
        && 160 == (((a << b) + c) << d)
        && 1073741824 == (a << ((b + (c << d)) - 5))
        && 512 == (a << b + c << d)
        ;
}

[numthreads(1, 1, 1)]
void computeMain(uint groupIndex : SV_GroupIndex, int3 dispatchThreadID: SV_DispatchThreadID)
{
    //BUF: 1
    bool result = true
        && Test_And_Or()
        && Test_And_Xor()
        && Test_Xor_Or()
        && Test_LShift_RShift()
        && Test_LShift_And()
        && Test_LShift_Or()
        && Test_LShift_Xor()
        && Test_Add_LShift()
        ;

    outputBuffer[0] = int(result);
}
