// Common type definitions for torch_vulkan shaders

// Push constants shared by most element-wise ops
struct ElementwiseParams {
    uint numel;
};

// Push constants for matrix operations
struct MatmulParams {
    uint M;
    uint N;
    uint K;
};
