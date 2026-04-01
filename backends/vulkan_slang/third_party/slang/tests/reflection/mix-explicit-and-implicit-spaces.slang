// mix-explicit-and-implicit-spaces.slang
//TEST:REFLECTION:-profile cs_5_1 -target hlsl -no-codegen

// Ensure that correct layout/reflection is computed
// when mixing implicit and explicit spaces for
// parameter blocks.

struct A { float x; }
ParameterBlock<A> a : register(space0);

struct B { float y; }
ParameterBlock<B> b : register(space1);

struct C { float z; }
ParameterBlock<C> c;

void main()
{}
