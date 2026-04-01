//TEST:COMPILE: tests/modules/environment.slang -o tests/modules/environment.slang-module
//TEST:COMPILE: tests/modules/hit.slang -stage closesthit -entry closesthit -target dxil -o monolithic.dxil

import environment;

StructuredBuffer <Environment_sample_data> environment_sample_buffer;

float3 sample_lights(inout uint seed)
{
    return environment_sample(environment_sample_buffer, seed);
}

struct RadianceHitInfo
{
    float3 contribution;
};

struct Attributes
{
    float2 bary;
};

[shader("closesthit")]
void closesthit(inout RadianceHitInfo payload, Attributes attrib) {}
