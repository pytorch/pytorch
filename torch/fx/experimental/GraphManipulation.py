def replace_all_uses_with(fx_module, old_op, old_target, new_op, new_target):
    """Modifies all nodes  in nodes which match the op code and target of old_node, and updates them to match new_node"""
    for node in fx_module.graph.nodes:
        if node.op == old_op and node.target == old_target:
            node.op = new_op
            node.target = new_target
    fx_module._generate_forward()
