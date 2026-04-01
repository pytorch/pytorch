// shaders.slang

//
// This example builds on the simplistic shaders presented in the
// "Hello, World" example by adding support for (intentionally
// simplistic) surface materil and light shading.
//
// The code here is not meant to exemplify state-of-the-art material
// and lighting techniques, but rather to show how a shader
// library can be developed in a modular fashion without reliance
// on the C preprocessor manual parameter-binding decorations.
//

// We are going to define a simple model for surface material shading.
//
// The first building block in our model will be the representation of
// the geometry attributes of a surface as fed into the material.
//
struct SurfaceGeometry
{
    float3 position;
    float3 normal;

    // TODO: tangent vectors would be the natural next thing to add here,
    // and would be required for anisotropic materials. However, the
    // simplistic model loading code we are currently using doesn't
    // produce tangents...
    //
    //      float3 tangentU;
    //      float3 tangentV;

    // We store a single UV parameterization in these geometry attributes.
    // A more complex renderer might need support for multiple UV sets,
    // and indeed it might choose to use interfaces and generics to capture
    // the different requirements that different materials impose on
    // the available surface attributes. We won't go to that kind of
    // trouble for such a simple example.
    //
    float2 uv;
};
//
// Next, we want to define the fundamental concept of a refletance
// function, so that we can use it as a building block for other
// parts of the system. This is a case where we are trying to
// show how a proper physically-based renderer (PBR) might
// decompose the problem using Slang, even though our simple
// example is *not* physically based.
//
interface IBRDF
{
    // Technically, a BRDF is only a function of the incident
    // (`wi`) and exitant (`wo`) directions, but for simplicity
    // we are passing in the surface normal (`N`) as well.
    //
    float3 evaluate(float3 wo, float3 wi, float3 N);
};
//
// We can now define various implemntations of the `IBRDF` interface
// that represent different reflectance functions we want to support.
// For now we keep things simple by defining about the simplest
// reflectance function we can think of: the Blinn-Phong reflectance
// model:
//
struct BlinnPhong : IBRDF
{
    // Blinn-Phong needs diffuse and specular reflectances, plus
    // a specular exponent value (which relates to "roughness"
    // in more modern physically-based models).
    //
    float3 kd;
    float3 ks;
    float specularity;

    // Here we implement the one requirement of the `IBRDF` interface
    // for our concrete implementation, using a textbook definition
    // of Blinng-Phong shading.
    //
    // Note: our "BRDF" definition here folds the N-dot-L term into
    // the evlauation of the reflectance function in case there are
    // useful algebraic simplifications this enables.
    //
    float3 evaluate(float3 V, float3 L, float3 N)
    {
        float nDotL = saturate(dot(N, L));
        float3 H = normalize(L + V);
        float nDotH = saturate(dot(N, H));

        // TODO: The current model loading has a bug that is leading
        // to the `ks` and `specularity` fields being invalid garbage
        // for our example cube, and the result is a non-finite value
        // coming out of `evaluate()` if we include the specular term.

        // return kd*nDotL + ks*pow(nDotH, specularity);
        return kd*nDotL;
    }
};
//
// It is important to note that a reflectance function is *not*
// a "material." In most cases, a material will have spatially-varying
// properties so that it cannot be summarized as a single `IBRDF`
// instance.
//
// Thus a "material" is a value that can produce a BRDF for any point
// on a surface (e.g., by sampling texture maps, etc.).
//
interface IMaterial
{
    // Different concrete material implementations might yield BRDF
    // values with different types. E.g., one material might yield
    // reflectance functions using `BlinnPhong` while another uses
    // a much more complicated/accurate representation.
    //
    // We encapsulate the choice of BRDF parameters/evaluation in
    // our material interface with an "associated type." In the
    // simplest terms, think of this as an interface requirement
    // that is a type, instead of a method.
    //
    // (If you are C++-minded, you might think of this as akin to
    // how every container provided an `iterator` type, but different
    // containers may have different types of iterators)
    //
    associatedtype BRDF : IBRDF;

    // For our simple example program, it is enough for a material to
    // be able to return a BRDF given a point on the surface.
    //
    // A more complex implementation of material shading might also
    // have the material return updated surface geometry to reflect
    // the result of normal mapping, occlusion mapping, etc. or
    // return an opacity/coverage value for partially transparent
    // surfaces.
    //
    BRDF prepare(SurfaceGeometry geometry);
};

// We will now define a trivial first implementation of the material
// interface, which uses our Blinn-Phong BRDF with uniform values
// for its parameters.
//
// Note that this implemetnation is being provided *after* the
// shader parameter `gMaterial` is declared, so that there is no
// assumption in the shader code that `gMaterial` will be plugged
// in using an instance of `SimpleMaterial`
//
struct SimpleMaterial : IMaterial
{
    // We declare the properties we need as fields of the material type.
    // When `SimpleMaterial` is used for `TMaterial` above, then
    // `gMaterial` will be a `ParameterBlock<SimpleMaterial>`, and these
    // parameters will be allocated to a constant buffer that is part of
    // that parameter block.
    //
    // TODO: A future version of this example will include texture parameters
    // here to show that they are declared just like simple uniforms.
    //
    float3 diffuseColor;
    float3 specularColor;
    float specularity;

    // To satisfy the requirements of the `IMaterial` interface, our
    // material type needs to provide a suitable `BRDF` type. We
    // do this by using a simple `typedef`, although a nested
    // `struct` type can also satisfy an associated type requirement.
    //
    // A future version of the Slang compiler may allow the "right"
    // associated type definition to be inferred from the signature
    // of the `prepare()` method below.
    //
    typedef BlinnPhong BRDF;

    BlinnPhong prepare(SurfaceGeometry geometry)
    {
        BlinnPhong brdf;
        brdf.kd = diffuseColor;
        brdf.ks = specularColor;
        brdf.specularity = specularity;
        return brdf;
    }
};
//
// Note that no other code in this file statically
// references the `SimpleMaterial` type, and instead
// it is up to the application to "plug in" this type,
// or another `IMaterial` implementation for the
// `TMaterial` parameter.
//

// A light, or an entire lighting *environment* is an object
// that can illuminate a surface using some BRDF implemented
// with our abstractions above.
//
interface ILightEnv
{
    // The `illuminate` method is intended to integrate incoming
    // illumination from this light (environment) incident at the
    // surface point given by `g` (which has the reflectance function
    // `brdf`) and reflected into the outgoing direction `wo`.
    //
    float3 illuminate<B:IBRDF>(SurfaceGeometry g, B brdf, float3 wo);
    //
    // Note that the `illuminate()` method is allowed as an interface
    // requirement in Slang even though it is a generic. Contrast that
    // with C++ where a `template` method cannot be `virtual`.
};

// Given the `ILightEnv` interface, we can write up almost textbook
// definition of directional and point lights.

struct DirectionalLight : ILightEnv
{
    float3 direction;
    float3 intensity;

    float3 illuminate<B:IBRDF>(SurfaceGeometry g, B brdf, float3 wo)
    {
        return intensity * brdf.evaluate(wo, direction, g.normal);
    }
};
struct PointLight : ILightEnv
{
    float3 position;
    float3 intensity;

    float3 illuminate<B:IBRDF>(SurfaceGeometry g, B brdf, float3 wo)
    {
        float3 delta = position - g.position;
        float d = length(delta);
        float3 direction = normalize(delta);
        float3 illuminance = intensity / (d*d);
        return illuminance * brdf.evaluate(wo, direction, g.normal);
    }
};

// In most cases, a shader entry point will only be specialized for a single
// material, but interesting rendering almost always needs multiple lights.
// For that reason we will next define types to represent *composite* lighting
// environment with multiple lights.
//
// A naive approach might be to have a single undifferntiated list of lights
// where any type of light may appear at any index, but this would lose all
// of the benefits of static specialization: we would have to perform dynamic
// branching to determine what kind of light is stored at each index.
//
// Instead, we will start with a type for *homogeneous* arrays of lights:
//
struct LightArray<L : ILightEnv, let N : int> : ILightEnv
{
    // The `LightArray` type has two generic parameters:
    //
    // - `L` is a type parameter, representing the type of lights that will be in our array
    // - `N` is a generic *value* parameter, representing the maximum number of lights allowed
    //
    // Slang's support for generic value parameters is currently experimental,
    // and the syntax might change.

    int count;
    L lights[N];

    float3 illuminate<B:IBRDF>(SurfaceGeometry g, B brdf, float3 wo)
    {
        // Our light array integrates illumination by naively summing
        // contributions from all the lights in the array (up to `count`).
        //
        // A more advanced renderer might try apply sampling techniques
        // to pick a subset of lights to sample.
        //
        float3 sum = 0;
        for( int ii = 0; ii < count; ++ii )
        {
            sum += lights[ii].illuminate(g, brdf, wo);
        }
        return sum;
    }
};

// `LightArray` can handle multiple lights as long as they have the
// same type, but we need a way to have a scene with multiple lights
// of different types *without* losing static specialization.
//
// The `LightPair<T,U>` type supports this in about the simplest way
// possible, by aggregating a light (environment) of type `T` and
// one of type `U`. Those light environments might themselves be
// `LightArray`s or `LightPair`s, so that arbitrarily complex
// environments can be created from just these two composite types.
//
// This is probably a good place to insert a reminder the Slang's
// generics are *not* C++ templates, so that the error messages
// produced when working with these types are in general reasonable,
// and this is *not* any form of "template metaprogramming."
//
// That said, we expect that future versions of Slang will make
// defining composite types light this a bit less cumbersome.
//
struct LightPair<T : ILightEnv, U : ILightEnv> : ILightEnv
{
    T first;
    U second;

    float3 illuminate<B:IBRDF>(SurfaceGeometry g, B brdf, float3 wo)
    {
        return first.illuminate(g, brdf, wo)
            + second.illuminate(g, brdf, wo);
    }
};

// As a final (degenerate) case, we will define a light
// environment with *no* lights, which contributes no illumination.
//
struct EmptyLightEnv : ILightEnv
{
    float3 illuminate<B:IBRDF>(SurfaceGeometry g, B brdf, float3 wo)
    {
        return 0;
    }
};

// The code above constitutes the "shader library" for our
// application, while the code below this point is the
// implementation of a simple forward rendering pass
// using that library.
//
// While the shader library has used many of Slang's advanced
// mechanisms, the vertex and fragment shaders will be
// much more modest, and hopefully easier to follow.


// We will start with a `struct` for per-view parameters that
// will be allocated into a `ParameterBlock`.
//
// As written, this isn't very different from using an HLSL
// `cbuffer` declaration, but importantly this code will
// continue to work if we add one or more resources (e.g.,
// an enironment map texture) to the `PerView` type.
//
struct PerView
{
    float4x4    viewProjection;
    float3      eyePosition;
};
ParameterBlock<PerView>     gViewParams;

// Declaring a block for per-model parameter data is
// similarly simple.
//
struct PerModel
{
    float4x4    modelTransform;
    float4x4    inverseTransposeModelTransform;
};
ParameterBlock<PerModel>    gModelParams;

// We want our shader to work with any kind of lighting environment
// - that is, and type that implements `ILightEnv`.

ILightEnv gLightEnv;

// Our handling of the material parameter for our shader
// is quite similar to the case for the lighting environment:
//
IMaterial gMaterial;

// Our vertex shader entry point is only marginally more
// complicated than the Hello World example. We will
// start by declaring the various "connector" `struct`s.
//
struct AssembledVertex
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : UV;
};
struct CoarseVertex
{
    float3 worldPosition;
    float3 worldNormal;
    float2 uv;
};
struct VertexStageOutput
{
    CoarseVertex    coarseVertex    : CoarseVertex;
    float4          sv_position     : SV_Position;
};

// Perhaps most interesting new feature of the entry
// point decalrations is that we use a `[shader(...)]`
// attribute (as introduced in HLSL Shader Model 6.x)
// in order to tag our entry points.
//
// This attribute informs the Slang compiler which
// functions are intended to be compiled as shader
// entry points (and what stage they target), so that
// the programmer no longer needs to specify the
// entry point name/stage through the API (or on
// the command line when using `slangc`).
//
// While HLSL added this feature only in newer versions,
// the Slang compiler supports this attribute across
// *all* targets, so that it is okay to use whether you
// want DXBC, DXIL, or SPIR-V output.
//
[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    VertexStageOutput output;

    float3 position = assembledVertex.position;
    float3 normal   = assembledVertex.normal;
    float2 uv       = assembledVertex.uv;

    float3 worldPosition = mul(gModelParams.modelTransform, float4(position, 1.0)).xyz;
    float3 worldNormal = mul(gModelParams.inverseTransposeModelTransform, float4(normal, 0.0)).xyz;

    output.coarseVertex.worldPosition = worldPosition;
    output.coarseVertex.worldNormal   = worldNormal;
    output.coarseVertex.uv            = uv;

    output.sv_position = mul(gViewParams.viewProjection, float4(worldPosition, 1.0));

    return output;
}

// Our fragment shader is almost trivial, with the most interesting
// thing being how it uses the `TMaterial` type parameter (through the
// value stored in the `gMaterial` parameter block) to dispatch to
// the correct implementation of the `getDiffuseColor()` method
// in the `IMaterial` interface.
//
// The `gMaterial` parameter block declaration thus serves not only
// to group certain shader parameters for efficient CPU-to-GPU
// communication, but also to select the code that will execute
// in specialized versions of the `fragmentMain` entry point.
//
[shader("fragment")]
float4 fragmentMain(
    CoarseVertex coarseVertex : CoarseVertex) : SV_Target
{
    // We start by using our interpolated vertex attributes
    // to construct the local surface geometry that we will
    // use for material evaluation.
    //
    SurfaceGeometry g;
    g.position = coarseVertex.worldPosition;
    g.normal = normalize(coarseVertex.worldNormal);
    g.uv = coarseVertex.uv;

    float3 V = normalize(gViewParams.eyePosition - g.position);

    // Next we prepare the material, which involves running
    // any "pattern generation" logic of the material (e.g.,
    // sampling and blending texture layers), to produce
    // a BRDF suitable for evaluating under illumination
    // from different light sources.
    //
    // Note that the return type here is `gMaterial.BRDF`,
    // which is the `BRDF` type *associated* with the (unknown)
    // `TMaterial` type. When `TMaterial` gets substituted for
    // a concrete type later (e.g., `SimpleMaterial`) this
    // will resolve to a concrete type too (e.g., `SimpleMaterial.BRDF`
    // which is an alias for `BlinnPhong`).
    //
    let brdf = gMaterial.prepare(g);

    // Now that we've done the first step of material evaluation
    // and sampled texture maps, etc., it is time to start
    // integrating incident light at our surface point.
    //
    // Because we've wrapped up the lighting environment as
    // a single (composite) object, this is as simple as calling
    // its `illuminate()` method. Our particular fragment shader
    // is thus abstracted from how the renderer chooses to structure
    // this integration step, somewhat similar to how an
    // `illuminance` loop in RenderMan Shading Language works.
    //

    float3 color = saturate(gLightEnv.illuminate(g, brdf, V) + float3(0.3));
    return float4(color, 1);
}
