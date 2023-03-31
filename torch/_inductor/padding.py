import torch

def needs_padding(args):
  
  for arg in args:
    size = arg.meta["val"].shape
    for i in size:
      if i % get_alignment_size(i) or i < 512:
        return False
    return True
  

def larger_closest_multiple(n, k):
    if n % k == 0:
        return n
    else:
        return n + k - (n % k)

def get_alignment_size(dtype):
    # try:
    #     gpu_name = torch.cuda.get_device_name()
    # except RuntimeError:
    #     gpu_name = ""
    # if "A100" in gpu_name:
    #     has_a100 = True
    # else:
    #     has_a100 = False
    if dtype == torch.int8:
        return 16
    elif dtype in [torch.float16, torch.half, torch.bfloat16]:
        return 8
    elif dtype == torch.float32:
        return 4
    elif dtype == torch.float64:
        return 2
    else:
        return 1

def pad_mm(fx_g: torch.fx.GraphModule):
    
    # Leverages a classic interpreter pattern, thanks Horace!
    new_graph = torch.fx.Graph()
    env = {}
    for node in fx_g.graph.nodes:
        if node.target == torch.ops.aten.addmm.default:
          
          # Currently this is a heuristic that decides if we should pad
          # Decided to only pad for medium size matrices and if alignment is off
          if needs_padding(node.args):
              size = int(tuple(env[node.args[0]].meta["tensor_meta"].shape)[0])
              alignment = get_alignment_size(env[node.args[0]].meta["tensor_meta"].dtype)
              pad_amount = larger_closest_multiple(size, alignment) - size

              # For each matmul, pad the matrix with zeroes, do the matmu and then slice to return a tensor size that the user expects
              new_a_pad = new_graph.call_function(torch.ops.aten.cat, (env[node.args[0]], torch.ops.aten.zeros([pad_amount, 1])))
              new_mm_pad = new_graph.call_function(torch.ops.aten.addmm.default, (new_a_pad, env[node.args[1]]))
              new_mm = new_graph.call_function(torch.ops.aten.slice, (new_mm_pad, 0, pad_amount))
          env[node] = new_mm
        
        else:
            new_node = new_graph.node_copy(node, lambda n: env[n])
            env[node] = new_node
    return torch.fx.GraphModule(fx_g, new_graph)