// pass-through-no-stage.hlsl

// Trying to compile in `-pass-through` mode without
// specifying a stage is an error, because the downstream
// compilers don't support inferring the stage from
// an attribute.

//DIAGNOSTIC_TEST:SIMPLE:-pass-through fxc -entry main
