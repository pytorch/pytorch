# Simple WebGPU example using Slang WebAssembly library

## Description

This is a simple example showing how WebGPU applications can use the slang-wasm library to compile slang shaders at runtime to WGSL.
The resulting application shows a green triangle rendered on a black background.

## Instructions

Follow the WebAssembly build instructions in `docs/building.md` to produce `slang-wasm.js` and `slang-wasm.wasm`, and place these files in this directory.

Start a web server, for example by running the following command in this directory:

    $ python -m http.server

Finally, visit `http://localhost:8000/` to see the application running in your browser.