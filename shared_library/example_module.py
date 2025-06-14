# shared_library/example_module.py

def add_numbers(a: float, b: float) -> float:
    """
    Adds two numbers and returns the result.
    This is a basic example function.
    """
    result = a + b
    return result

class SimpleLayer:
    """
    A very simple example of a layer class that might be part of the shared library.
    In a real scenario, this would have more complex logic, perhaps similar to a
    PyTorch nn.Module.
    """
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        # In a real layer, you might initialize weights here
        # self.weights = initialize_weights(input_size, output_size)
        print(f"SimpleLayer initialized with input_size={input_size}, output_size={output_size}")

    def forward(self, input_data):
        """
        A placeholder for the forward pass of the layer.
        """
        print(f"SimpleLayer: forward pass with input_data of shape/type: {type(input_data)}")
        # In a real layer, you'd perform computations using input_data and self.weights
        # For this example, let's assume it returns a transformed output of the correct size.
        # This is highly simplified.
        if hasattr(input_data, 'shape') and len(input_data.shape) > 0:
             # Crude example: if input is like a batch, create output of [batch_size, self.output_size]
            output_shape = list(input_data.shape)
            output_shape[-1] = self.output_size
            # This is just a placeholder for actual computation
            # In a real scenario, you'd use something like numpy or torch zeros/random
            output_data = [[0.0] * self.output_size for _ in range(output_shape[0])]
            print(f"SimpleLayer: simulated output data of shape: ({output_shape[0]}, {self.output_size})")
            return output_data
        else:
            # Fallback for non-array like input, or simple scalar
            output_data = [0.0] * self.output_size
            print(f"SimpleLayer: simulated output data for scalar/unknown input: {output_data}")
            return output_data


    def __repr__(self):
        return f"SimpleLayer(input_size={self.input_size}, output_size={self.output_size})"

if __name__ == '__main__':
    # Example usage of the components in this module
    sum_result = add_numbers(5, 3)
    print(f"Result of add_numbers(5, 3): {sum_result}")

    layer = SimpleLayer(input_size=10, output_size=5)
    print(f"Created layer: {layer}")

    # Simulate some input data (e.g., a batch of 2 samples, 10 features each)
    example_input = [[float(i) for i in range(10)] for _ in range(2)]
    output = layer.forward(example_input)
    print(f"Output of layer.forward(example_input): {output}")

    example_scalar_input = 5.0 # Less common for NN layers but for testing
    scalar_output = layer.forward(example_scalar_input)
    print(f"Output of layer.forward(example_scalar_input): {scalar_output}")
