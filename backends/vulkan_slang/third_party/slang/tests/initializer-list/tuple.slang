//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type

Tuple<bool, float> createTuple() {
    return {};
}

// We should also enable the following use of initialization list:

Tuple<bool, float> createTuple2() {
    return {false, 1.0};
}


//TEST_INPUT: set output = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> output;

[numthreads(1, 1, 1)]
void computeMain()
{
    let hit = createTuple();
    output[0] = hit._1 + 1.0;

    let hit2 = createTuple2();
    output[1] = hit2._1 + 1.0;
    // CHECK: 1.0
    // CHECK: 2.0
}