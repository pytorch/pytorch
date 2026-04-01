Binding Generation Tests
========================

These tests ensure that the compiler can correctly add explicit binding information (e.g., HLSL `register` semantics) to code that does not originally have them.

Example
-------

Given code like:

    Texture2D ta;
    Texture2D tb;

We expect to produce output like:

    Texture2D ta : register(t0);
    Texture2D tb : register(t1);

The resulting code guarantees that `tb` will always be assigned to the same location, regardless of how these values are (or are not) used in later shader code.

Methodology
-----------

These tests currently rely on the ability to run the same HLSL code through the Slang compiler driver and execute either Slang, or HLSL. We write an example like the above by wrapping explicit `register` semantics in a macro:

    Texture2D ta R(: register(t0));
    Texture2D tb R(: register(t1));

In the HLSL case, these annotations will manually place things where we want them, while in the Slang case, we define the macro to have an empty expansion, so that the annotations express our expectation for what the compiler will auto-generate.