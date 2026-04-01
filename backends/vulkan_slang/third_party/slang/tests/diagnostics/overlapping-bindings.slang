// overlapping-bindings.slang

//DIAGNOSTIC_TEST:SIMPLE:-target hlsl

// Two parameters with the same `register:

Texture2D a : register(t0);

Texture2D b : register(t0);

// Parameters marked to ignore overlap:

[allow("overlapping-bindings")]
Texture2D c : register(t1);

[allow("overlapping-bindings")]
Texture2D d : register(t1);
