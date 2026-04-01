// example-effect.slang

// This file provides an example of how a shader
// toy effect can be compiled as a Slang module.
//
// Every effect will depend on the module that
// defines out shader toy infrastructure:
//
import shader_toy;

// The `shader_toy` module defines the interface
// that each effect must implement, and our
// specific effect will be a type that implements
// the interface:
//
struct ExampleEffect : IShaderToyImageShader
{
    // Our goal is that we can mostly just copy-paste
    // the code for an effect from shadertoy.com into
    // this file, and have something that works.
    //
    // Due to limitations in compatibility between
    // GLSL and Slang, that won't always work, but
    // it still helps to note where the original
    // GLSL code begins/ends.

    // Note: the verison of this file that is checked
    // in uses a placeholder effect so that this file
    // does not need to concern itself with the license
    // terms of particular effects on shadertoy.com.

// BEGIN GLSL

float rand(float n)
{
    return fract(sin(n) * 43758.5453123);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    float screenScale = length(iResolution.xy);
    vec2 uv = fragCoord / screenScale;

    float frequency = 5.0f;
    vec2 pos = (uv + iTime*vec2(0.25f, 0.0f)) * frequency;

    vec2 center = floor(pos + vec2(0.5));

    float r0 = rand(center.x*3.0f  + center.y*7.0f);
    float r1 = rand(center.x*7.0f  + center.y*13.0f);
    float r2 = rand(center.x*13.0f + center.y*3.0f);

    float p = mix(0.0f, 4.0f, r0);
    float f = mix(5.0f, 8.0f, r1);

    float a = 0.5f * (1.0f + cos(iTime*f + p));

    float rad0 = mix(0.1, 0.4, r2);
    float rad1 = mix(0.2, 0.9, r0);

    float radius = 0.5f*mix(rad0, rad1, a);

    vec2 delta = pos - center;
    float distance = length(delta);

    fragColor.xyz = vec3(r0, r1, r2);
    fragColor.w = 1.0f;

    if(distance > radius) fragColor.xyz = vec3(0.25f);
}

// END GLSL

    // The GLSL logic for the effect above might have included
    // "global" declarations (which become fields since things
    // are wrapped in a `struct`) with initializer, and we
    // need a way for the code that uses an effect like this
    // to get an instance that has been properly initialized.
    //
    // Right now, the `IShaderToyImageShader` interface requires
    // a factory function `getDefault()`, so we will implement
    // that here.
    //
    static This getDefault()
    {
        // Note: this code does not need to be updated for different
        // GLSL effects, since it will default-initialize whatever
        // members the `This` types has.
        //
        This value = {};
        return value;
    }
}
