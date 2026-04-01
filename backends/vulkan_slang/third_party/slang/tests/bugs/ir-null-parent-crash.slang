// ir-null-parent-crash.slang

// Test an issue where Slang was crashing on functions that
// have `static` variables with initializers.

//TEST:SIMPLE:-target hlsl

struct RNG
{
	uint state[2];
}

void jump(inout RNG rng)
{
    static uint32_t a[] = { 0x4, 0x2, 0x1, 0x3 };

    uint32_t s0 = 0;

    for (int i = 0; i < 4; i++)
    {
        s0 ^= rng.state[0];
    }

    rng.state[0] = s0;
}
