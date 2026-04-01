// shader-toy.slang

// This file implements the core of a system for executing
// code from shadertoy.com in the context of Slang.
//
// The big idea here is to define an interface so that
// different shader toy effects can be defined as
// separately compiled modules, and then "plug in" to
// execution environment for running those effects, wheter
// via vertex/fragment shaders, compute, or on CPU.
//
// An important goal is that we should be able to run effects
// defined on shadertoy.com with as little modification as
// possible. This goal isn't 100% achievable because shader
// toy effects are authored in GLSL, which differs from Slang
// in several ways, so this is an aspirational goal rather
// than a requirement.
//
// There are a few different kinds of effects supported
// by shader toy, which are enumerated on the [How To](https://www.shadertoy.com/howto)
// page of the project. This module focuses only on
// "image shaders," which are by far the most common.
//
// We will start with the interface that all image shaders
// are expected to implement.
//
interface IShaderToyImageShader
{
    // The shader toy "How To" page says:
    //
    // > Image shaders implement the `mainImage()` function in order
    // > to generate procedural images by computing a color for
    // > each pixel. ...
    //
    // The GLSL signature given is:
    //
    // > void mainImage( out vec4 fragColor, in vec2 fragCoord );
    //
    // We can translate that signature almost verbatim into Slang:
    //
    void mainImage( out float4 fragColor, in float2 fragCoord );

    // An image shader effect will thus be a Slang `struct` type
    // that implements the `IShaderToyImageShader` interface.
    // In most cases, effects will be created by pasting the
    // GLSL content of an effect into boilerplate `struct`
    // definition.
    //
    // As a result of scoping the GLSL effect code in a Slang
    // `struct`, any functions declared in the effect will become
    // methods of the struct, and any global-scope variables declared
    // in the effect will become members of the `struct` type.
    //
    // Note: One caveat that arises here is that any effect that
    // makes use of mutable global variables in its GLSL code will
    // fail to compile with our approach. By default methods in
    // Slang have can only read from `this` and its members, and
    // need to be marked a `[mutating]` to have read-write access.
    // Most shader toy effects only use globals to declare constants,
    // so this limitation may not be a big problem in practice.
    //
    // One thing that *does* matter is that global variables/constants in
    // an effect may have initializers, and the behavior of the
    // effect is likely to depend on them being initialized correctly.
    //
    // In practice, the need to have effect-specific initialization
    // is another requirement of the image shader interface that doesn't
    // need to be explicitly explained on shadertoy.com, but does matter
    // when writing an explicit interface in Slang.
    //
    // We express the requirement using a `static` method that
    // returns the `This` type. Much like `this` refers to the
    // "current object" with whatever type it might have at runtime,
    // the `This` type refers to the "current type" that is implementing
    // this interface. Static methods that return `This` allow
    // Slang to express required "factory" functions in an interface.
    //
    static This getDefault();
};

// Now that we have defined the interface that all image
// shader effects are expected to implement, we can define
// a vertex and fragment shader that can be used to evaluate
// any effect that conforms to the interface.
//
// The vertex shader is just going to implement a full-screen
// triangle, so it is almost trivial:
//
[shader("vertex")]
float4 vertexMain(float2 position : POSITION)
    : SV_Position
{
    // TODO: We could even turn this into a shader that
    // takes no inputs, and directly computes the XY
    // location of each vertex based on `SV_VertexID`

    return float4(position, 0.5, 1.0);
}
//
// The body of the effect will run in the fragment shader,
// so that is where things get more interesting.
//
// We will define our fragment shader entry point as a
// Slang *generic function*, with a generic type parameter
// `T` that is constrained to implement the `IShaderToyImageShader`
// interface.
//
[shader("fragment")]
float4 fragmentMain<T : IShaderToyImageShader>(
    float4 sv_position : SV_Position)
    : SV_Target
{
    // Because the Slang compiler knows the interface that `T`
    // is expected to implement, any code in the function
    // body will be checked to make sure that it does not
    // use operations that `T` would not be guaranteed to
    // support. Unlike with traditional C++ templates or
    // preprocessor-based shader specialization, it is possible
    // to copmile and type-check this entry point once,
    // and use it with multiple different types for `T`.

    // We start by creating an instance of the effect
    // type `T`, initialized to whatever its default
    // values are.
    //
    // Note: initializing the effect here preserves the
    // semantics of the original shader toy code. An
    // alternative approach (that would change the behavior
    // from the original) would be to pass a value of
    // type `T` in as a shader parameter of the entry point.
    //
    T toy = T.getDefault();

    // Next, we invoke the user-defined effect by calling
    // its `mainImage` function.
    //
    // Recall that the `fragColor` parameter to `mainImage`
    // was defined as an `out` parameter, so this call
    // will set a value into our local `fragColor` variable.
    //
    float2 fragCoord = sv_position.xy;
    float4 fragColor = 0;
    toy.mainImage(fragColor, fragCoord);

    // The output value from our shader is simply the result
    // from the user-defined effect.
    //
    return float4(fragColor.xyz, 1);
}

// By defining an interface for image shader effects, we
// have been able to decouple the code for the effects
// themselves from the code for their execution contexts.
// A key benefit of that decoupling is that we can introduce
// both new effects and new execution contexts in a modular
// fashion.
//
// For example, we can easily define a compute shader for
// executing an image shader effect:
//
[shader("compute")]
void computeMain<T : IShaderToyImageShader>(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform RWTexture2D<float4> image)
{
    // The operations required to set up and execute
    // the user-defined effect are similar to what
    // they were for the fragment shader.
    //
    T toy = T.getDefault();

    float2 fragCoord = float2(sv_dispatchThreadID.xy);
    float4 fragColor = 0;
    toy.mainImage(fragColor, fragCoord);

    // The main difference is that we now write the
    // output color explicitly to an image pixel
    // instead of relying on the rasterization pipeline.
    //
    image[sv_dispatchThreadID.xy] = fragColor;
}

// At this point we've described how our module will
// execute shader toy effects that implement the
// required interface, but we also need to set up
// the services that those effects are able to use.
//
// The shader toy "How To" file describes a large number of uniform
// shader parameters that are implicitly visible to every effect.
//
// If we were able to design an interface from scratch, we might
// prefer to make the `mainImage` function take some kind of
// explicit context parameter that provides access to these
// values, but because our goal is to be compatible with existing
// effects with their established `mainImage` signature, we will
// instead define these parameters using old-fashioned global-scope
// shader parameters.
//
cbuffer ShaderToyUniforms
{
    // Note: We do not currently define all of the parameters
    // exposed by Shader Toy, but rather just the most commonly
    // used ones.
    //
    // TODO: We can and should fill in the rest over time.
    //
    float4 iMouse;
    float2 iResolution;
    float iTime;
};

// In addition to the above parameters that use ordinary data types,
// shader toy also exposes the `iChannel*` parameters (`iChannel0`
// through `iChannel4`). These parameters represent sampled image
// inputs that can be bound to selected images as part of an effect.
//
// Traditional GLSL "sampler" types include both the texture image
// and sampler state, while Slang (like D3D, Vulkan, etc.) has
// distinct texture and sampler types. In order to define the
// channel variables in a way that is compatible with shader toy,
// we will define a `struct` type for a pair of a texture and
// a sampler:
//
struct TextureSamplerPair
{
    Texture2D t;
    SamplerState s;
};

// With our texture-sampler pair type defined, we can introduce
// the variables for the texture channels easily.
//
TextureSamplerPair iChannel0;
TextureSamplerPair iChannel1;
TextureSamplerPair iChannel2;
TextureSamplerPair iChannel3;


// TODO: Shader toy supports more than just 2D textures, so a good
// avenue for extension of the module would be to define an interface
// for the texture channels, and have implementations using various
// forms of textures.
//
// A really ambitious idea would be include one example of the
// texture-channel interface that uses an existing ShaderToy as a
// procedural texture.

// Shader toy effects access the contents of the `iChannel*` variables
// using the `texture()` function, so we need to provide a definition
// that is suitable:
//
float4 texture(TextureSamplerPair p, float3 uvw)
{
    // TODO: The right implementation to use here (at least
    // in the context of fragment shaders) is:
    //
    //      return p.t.Sample(p.s, uvw.xy);
    //
    // However, the current implementation of the main
    // application code doesn't include texture image
    // loading, so we will instead just fill in a
    // placeholder result for "texture" lookup:
    //
    return 0.5;
}

// The last major issue we need to address in this module is the way that
// shader toy effects are authored in GLSL, which has several differences
// from Slang that could cause problems.
//
// Some of these differences can be surmounted relatively easily. For
// example, GLSL uses different names for its built-in vector types, but
// for the most part they are compatible with those defined by HLSL/Slang.
// We can paper over this difference by defining a few helpful type
// aliases.
//
typealias vec2 = float2;
typealias vec3 = float3;
typealias vec4 = float4;

// Matrix types in GLSL are a more subtle issue, because they have different
// semantics from their HLSL/Slang equivalents in a few key ways:
//
// * The infix `*` operator always performs component-wise multiplication
//   in HLSL/Slang, but in GLSL it sometimes performs linear-algebraic
//   products (whenever we have matrix*matrix, vector*matrix, or matrix*vector).
//   HLSL/Slang require a distinct `mul()` function for those cases.
//
// * Because of differences in terminology and conventions, a linear-algebraic
//   product like `M*v` in GLSL is equivalent to `mul(v, M)` in HLSL/Slang
//   (note the reversed order of operands).
//
// * Constructing a matrix or vector from a single scalar consistently
//   replicates that scalar across all components/elements in HLSL/Slang,
//   but in GLSL it instead produces a diagonal matrix.
//
// These differences are not something we can surmount by defining the
// GLSL matrix types as aliases of the standard Slang ones, so instead we
// must define the GLSL matrix types as wrappers around the Slang ones.
//
struct mat2
{
    float2x2 data;

    __init(float e00, float e01, float e10, float e11)
    {
        data = float2x2(e00, e01, e10, e11);
    }

    // TODO: We need to fill in the other intializers and members
    // available on matrices here.
};

// TODO: fill in `mat3` and `mat4`.

// TODO: Ideally we would want to define overloaded operation functions
// to allow `*`, `*=`, etc. to apply to our user-space matrix types.
//
// Unfortunately, implementation bugs in the Slang compiler mean that
// user-defined operator overloads aren't working right now.
//
//      vec2 operator*(vec2 left, mat2 right)
//      {
//          return mul(right.data, left);
//      }
//
//      void operator*=(inout vec2 left, mat2 right)
//      {
//          left = mul(right.data, left);
//      }
//
// Instead, we will define an ordinary function for the one case that
// we've run into in a test effect so far:
//
void mulAssign(inout vec2 left, mat2 right)
{
    left = mul(right.data, left);
}

float fract(float value)
{
    return frac(value);
}

float mix(float a, float b, float t)
{
    return lerp(a, b, t);
}
