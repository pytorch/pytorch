import torch


@torch.jit.script
def foo(x, y):
    out = x.mm(y)
    return [out + 1]


torch._C._jit_pass_inline(foo.graph)

all_fusion_sequences = []

def gen_fusion_opportunities(block):
    pointwise_ops = []
    for curr_node in block.nodes():
        if curr_node.isMatmulOp():
            chain = [curr_node]
            size = 1
            if "addmm" in str(curr_node.kind()):
                size += 1
            queue = []
            def add_output_nodes(n):
                for output_v in n.outputs():
                    for use in output_v.uses():
                        if use.user.isPointwiseOP():
                            queue.append(use.user)
            add_output_nodes(curr_node)
            seen = set()
            while queue:
                node = queue[0]
                queue = queue[1:]
                if (not node in seen):
                    seen.add(node)
                    chain.append(node)
                    add_output_nodes(node)
            all_fusion_sequences.append(chain)
        for b in curr_node.blocks():
            gen_fusion_opportunities(b)

gen_fusion_opportunities(foo.graph)
print(all_fusion_sequences)
