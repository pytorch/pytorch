//TEST_IGNORE_FILE:

// Companion file to `gh-38-fs.hlsl`

Texture2D overlappingB : register(t0);

Texture2D conflicting : register(t2);

float4 main() : SV_Target { return 0; }
