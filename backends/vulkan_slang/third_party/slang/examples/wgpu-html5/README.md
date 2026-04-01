# Simple WebGPU example

## Description

This is a simple example showing how WebGPU and Slang can be used together.
The resulting application shows a green triangle rendered on a black background.

More serious applications are adviced to make use of Slang's reflection API.

## Instructions

Get `slangc` from https://github.com/shader-slang/slang/releases/latest, or build it using the instructions under `docs/building.md`, and make sure that `slangc` is in [the `PATH` of your shell](https://en.wikipedia.org/wiki/PATH_(variable)).

Compile the Slang shaders `shader.slang` into WGSL shaders named `shader.vertex.wgsl` and `shader.fragment.wgsl`:

    $ slangc -target wgsl -stage vertex -entry vertexMain -o shader.vertex.wgsl shader.slang
    $ slangc -target wgsl -stage fragment -entry fragmentMain -o shader.fragment.wgsl shader.slang

Alternatively, you can run `build.py` which does the same thing.

Start a web server, for example by running the following command in this directory:

    $ python -m http.server

Finally, visit `http://localhost:8000/` to see the application running in your browser.