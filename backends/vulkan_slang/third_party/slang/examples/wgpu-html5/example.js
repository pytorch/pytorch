"use strict";

let Example = {
    initialize: async function (canvas) {
        async function render(shaders) {
            if (!navigator.gpu) {
                throw new Error("WebGPU not supported on this browser.");
            }
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                throw new Error("No appropriate GPUAdapter found.");
            }
            const device = await adapter.requestDevice();
            const context = canvas.getContext("webgpu");
            const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
            context.configure({
                device: device,
                format: canvasFormat,
            });

            const vertexBufferLayout = {
                arrayStride: 8,
                attributes: [{
                    format: "float32x2",
                    offset: 0,
                    shaderLocation: 0,
                }],
            };

            const pipeline = device.createRenderPipeline({
                label: "Pipeline",
                layout: "auto",
                vertex: {
                    module: device.createShaderModule({
                        label: "Vertex shader module",
                        code: shaders.vertex
                    }),
                    entryPoint: "vertexMain",
                    buffers: [vertexBufferLayout]
                },
                fragment: {
                    module: device.createShaderModule({
                        label: "Fragment shader module",
                        code: shaders.fragment
                    }),
                    entryPoint: "fragmentMain",
                    targets: [{
                        format: canvasFormat
                    }]
                }
            });

            const vertices = new Float32Array([
                0.0, -0.8,
                +0.8, +0.8,
                -0.8, +0.8,
            ]);
            const vertexBuffer = device.createBuffer({
                label: "Triangle vertices",
                size: vertices.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
            const bufferOffset = 0;
            device.queue.writeBuffer(vertexBuffer, bufferOffset, vertices);

            const encoder = device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: context.getCurrentTexture().createView(),
                    loadOp: "clear",
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    storeOp: "store",
                }]
            });
            pass.setPipeline(pipeline);
            const vertexBufferSlot = 0;
            pass.setVertexBuffer(vertexBufferSlot, vertexBuffer);
            pass.draw(vertices.length / 2);
            pass.end();
            const commandBuffer = encoder.finish();
            device.queue.submit([commandBuffer]);
        }

        render({
            vertex: await fetch("shader.vertex.wgsl").then(r => r.text()),
            fragment: await fetch("shader.fragment.wgsl").then(r => r.text()),
        });
    }
}
