//TEST:SIMPLE(filecheck=METAL): -stage compute -entry computeMain -target metal
//TEST:REFLECTION(filecheck=REFLECT): -stage compute -entry computeMain  -target metal

// ------------------------
Sampler2D sampler;

// METAL-DAG: {{sampler_texture_[1-9]}} {{\[\[}}texture(0)]]
// METAL-DAG: {{sampler_sampler_[1-9]}} {{\[\[}}sampler(0)]]

// REFLECT: "name": "sampler"
// REFLECT-NOT:}
// REFLECT: {"kind": "shaderResource", "index": 0}
// REFLECT-NOT:}
// REFLECT: {"kind": "samplerState", "index": 0}

// ------------------------
Texture2D followingTexture;

// METAL-DAG: {{followingTexture_[1-9]}} {{\[\[}}texture(1)]]

// REFLECT: "name": "followingTexture"
// REFLECT-NOT:}
// REFLECT: "binding": {"kind": "shaderResource", "index": 1}

// ------------------------
SamplerState followingSamplerState;

// METAL-DAG: {{followingSamplerState_[1-9]}} {{\[\[}}sampler(1)]]

// REFLECT: "name": "followingSamplerState"
// REFLECT-NOT:}
// REFLECT: "binding": {"kind": "samplerState", "index": 1}

// ------------------------
Sampler2D explicitBindingSampler : register(t4): register(s3);

// METAL-DAG: {{explicitBindingSampler_texture_[1-9]}} {{\[\[}}texture(4)]]
// METAL-DAG: {{explicitBindingSampler_sampler_[1-9]}} {{\[\[}}sampler(3)]]

// REFLECT: "name": "explicitBindingSampler"
// REFLECT-NOT:}
// REFLECT: {"kind": "shaderResource", "index": 4}
// REFLECT-NOT:}
// REFLECT: {"kind": "samplerState", "index": 3}

float4 computeMain() : SV_Target
{
    return sampler.Sample(float2(0, 0)) + explicitBindingSampler.Sample(float2(0, 0)) + followingTexture.Sample(followingSamplerState, float2(0,0));
}
