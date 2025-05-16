import torch
import os
import copy
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)
import torch.multiprocessing as mp
import torch.distributed as dist

def capture(fn):
    def inner(*args):
        gm = None
        actual_args = None
        kwargs = None

        def backend(gm_, args_, **kwargs_):
            nonlocal gm
            nonlocal actual_args
            nonlocal kwargs
            gm = gm_
            actual_args = args_
            kwargs = kwargs_
            return gm

        _ = torch.compile(fn, fullgraph=True, backend=backend)(*args)
        return gm, actual_args, kwargs

    return inner

class Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wq = torch.nn.Linear(16, 16)
        self.wk = torch.nn.Linear(16, 16)
        self.wv = torch.nn.Linear(16, 16)
        self.wo = torch.nn.Linear(16, 16)

    def forward(self, x):
        # make rank 0 graph slightly different
        # TODO: we're going to need to avoid 0-1 specialization,
        # but still not error on these conditionals (we need a hint)
        if torch.distributed.get_rank() >= 3:
            x = x + 1
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        # fake attention:
        xo = xq + xk + xv
        #return xq, xk, xv, xo, self.wo(xo)
        return self.wo(xo),

class TransformerBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Attention()

    def forward(self, x):
        return self.attn(x)

class Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = TransformerBlock()

    def forward(self, input):
        return self.block(input)

def distribute_model(model, world_size):
    tp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))

    # apply sequence parallel
    parallel_plan = {
        "attn": PrepareModuleInput(
            input_layouts=Shard(0), desired_input_layouts=Replicate()
        ),
        "attn.wq": ColwiseParallel(use_local_output=True),
        "attn.wk": ColwiseParallel(use_local_output=True),
        "attn.wv": ColwiseParallel(use_local_output=True),
        "attn.wo": RowwiseParallel(output_layouts=Shard(0)),
    }

    parallelize_module(
        module=model.block,
        device_mesh=tp_mesh,
        parallelize_plan=parallel_plan,
    )

def precompile(model, args, dir_path, world_size):
    # init fake process group
    fake_store = FakeStore()
    dist.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )
    distribute_model(model, world_size)
    # precompile
    gm, args, kwargs = capture(model)(*args)
    compiled_artifact = torch._inductor.standalone_compile(gm, args, options={'use_symbolic_rank': True})
    compiled_artifact.save(path=dir_path, format='unpacked')
    # destroy fake process group
    dist.destroy_process_group()


def dist_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # or your master node's IP
    os.environ['MASTER_PORT'] = '29500'      # use a free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def get_updated_args(model, args):
    import torch.utils._pytree as pytree
    params = {
        **dict(model.named_parameters(remove_duplicate=False)),
        **dict(model.named_buffers(remove_duplicate=False)),
    }

    params_flat, _ = pytree.tree_flatten(params)

    # There is a problem here:
    # the calling convention is whatever order dynamo
    # chose to lift params into extra inputs.
    # Can we enforce an invariant that this ordering respected here?
    full_args = []
    # first params
    full_args.extend(params_flat)
    # then normal forward args
    full_args.extend(args)
    # then current rank
    full_args.append(torch.distributed.get_rank())
    return full_args

def launch_precompiled_worker(rank, model, args, artifact_path, world_size):
    dist_setup(rank, world_size)
    try:
        model.to(rank)

        args = [a.to(rank) for a in args]
        distribute_model(model, world_size)

        loaded = torch._inductor.CompiledArtifact.load(path=artifact_path, format="unpacked", model=model)
        # TODO: calling convention changes from dynamo need to be recorded somewhere
        # so we don't have to hardcode them
        all_args = get_updated_args(model, args)

        compiled_outs = loaded(*all_args)
        assert isinstance(compiled_outs, list)
        compiled_outs[-1].sum().backward()
        compiled_grads = [p.grad.clone() for p in model.parameters()]

        for p in model.parameters():
            p.grad = None

        eager_outs = model(*args)
        eager_outs[-1].sum().backward()
        eager_grads = [p.grad.clone() for p in model.parameters()]

        print("fw outs")
        for eager_out, compiled_out in zip(eager_outs, compiled_outs):
            print(torch.allclose(eager_out, compiled_out))
        print("grads")
        for eager_grad, compiled_grad in zip(eager_grads, compiled_grads):
            assert isinstance(eager_grad, DTensor)
            assert isinstance(compiled_grad, DTensor)
            print(torch.allclose(eager_grad._local_tensor, compiled_grad._local_tensor))
    finally:
        dist.destroy_process_group()

def run_distributed_job(model, args, artifact_path, world_size):
    mp.spawn(launch_precompiled_worker, args=(model, args, artifact_path, world_size,), nprocs=world_size, join=True, daemon=True)

if __name__ == '__main__':
    args = (torch.rand(20, 16).to('cuda'),)
    model = Transformer().to('cuda')
    world_size = 2

    artifact_path = 'tmp_cache_dir'
    # compile if we have not already pre-compiled and cached to disk
    if True or not os.listdir(artifact_path):
        # clone model because compilation requires DTensor-ifying the passed-in model
        model_copy = copy.deepcopy(model)
        precompile(model_copy, args, artifact_path, world_size)
        assert os.listdir(artifact_path)

    # now run the precompiled distributed job
    run_distributed_job(model, args, artifact_path=artifact_path, world_size=world_size)



