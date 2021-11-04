import graphviz  # type: ignore[import]

def get_layer_name_type(layer):
    return "\n".join(f"{i}" for i in [layer.name, layer.type])

def trt_network_to_dot_graph(network):
    dot = graphviz.Digraph(comment="Network")

    # add nodes (layers)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        dot.node(get_layer_name_type(layer))

    # add nodes (inputs)
    for i in range(network.num_inputs):
        dot.node(network.get_input(i).name)

    # add nodes (outputs)
    for i in range(network.num_outputs):
        dot.node(network.get_output(i).name)

    # add layer->layer edges
    for a in range(network.num_layers):
        layer_a = network.get_layer(a)

        for b in range(network.num_layers):
            layer_b = network.get_layer(b)

            for i in range(layer_a.num_outputs):
                output_i = layer_a.get_output(i)

                for j in range(layer_b.num_inputs):
                    input_j = layer_b.get_input(j)

                    if output_i == input_j:
                        dot.edge(get_layer_name_type(layer_a), get_layer_name_type(layer_b), label=str(input_j.shape))

    # add input->layer edges
    for i in range(network.num_inputs):
        input_i = network.get_input(i)

        for b in range(network.num_layers):
            layer_b = network.get_layer(b)

            for j in range(layer_b.num_inputs):
                input_j = layer_b.get_input(j)

                if input_i == input_j:
                    dot.edge(input_i.name, get_layer_name_type(layer_b), label=str(input_j.shape))

    # add layer->output edges
    for i in range(network.num_outputs):
        input_i = network.get_output(i)

        for b in range(network.num_layers):
            layer_b = network.get_layer(b)

            for j in range(layer_b.num_outputs):
                input_j = layer_b.get_output(j)

                if input_i == input_j:
                    dot.edge(get_layer_name_type(layer_b), input_i.name, label=str(input_j.shape))

    return dot
