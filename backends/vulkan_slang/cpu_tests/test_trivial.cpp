// Trivial CPU test — validates build system works
// When Slang CPU compilation is available, this will test shader math

#include <cassert>
#include <cmath>
#include <cstdio>

// Simple reference implementations to test against Slang CPU output
float ref_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float ref_gelu(float x) {
    const float k = 0.7978845608f;
    const float c = 0.044715f;
    float inner = k * (x + c * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

void test_relu() {
    float inputs[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    float expected[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f, 2.0f};

    for (int i = 0; i < 7; i++) {
        float result = ref_relu(inputs[i]);
        assert(fabsf(result - expected[i]) < 1e-6f);
    }
    printf("  PASS: relu\n");
}

void test_gelu() {
    float inputs[] = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
    for (float x : inputs) {
        float result = ref_gelu(x);
        // Just verify it's finite and reasonable
        assert(std::isfinite(result));
        assert(fabsf(result) < 10.0f);
    }
    // GELU(0) should be 0
    assert(fabsf(ref_gelu(0.0f)) < 1e-6f);
    printf("  PASS: gelu\n");
}

int main() {
    printf("Running CPU shader math tests...\n");
    test_relu();
    test_gelu();
    printf("All CPU tests passed!\n");
    return 0;
}
