"""Capture the actual strict Inductor kernel and exhaustive BF16 outputs."""

import importlib.metadata
import json
import os
from pathlib import Path
import sys

import torch
import triton


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


def main():
    assert torch.cuda.get_device_capability(0) == (8, 9)
    expected = os.environ['PIN_PROBE_SHA']
    assert expected[:8] in importlib.metadata.version('triton')
    site = os.environ.get('PIN_PROBE_SITE')
    if site:
        assert Path(triton.__file__).resolve().is_relative_to(site)
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    original = triton.compile

    def capture(*args, **kwargs):
        kernel = original(*args, **kwargs)
        prefix = output / (kernel.name + '-' + str(len(list(output.glob('*.ptx')))))
        for stage in ('ttir', 'ttgir', 'llir', 'ptx', 'cubin'):
            value = kernel.asm[stage]
            path = prefix.with_suffix('.' + stage)
            path.write_bytes(value) if isinstance(value, bytes) else path.write_text(value)
        source = args[0]
        prefix.with_suffix('.py').write_text(source.fn.src)
        prefix.with_suffix('.options.json').write_text(json.dumps(serializable({
            'kwargs': kwargs, 'constants': source.constants,
            'signature': source.signature, 'attrs': source.attrs,
        }), indent=2))
        print('ACTUAL_STRICT_ABS_CODEGEN', kernel.name, kwargs, flush=True)
        return kernel

    triton.compile = capture
    torch._inductor.config.compile_threads = 1
    x = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    eager = x.abs()
    compiled = torch.compile(torch.abs, options={
        'numerics': 'strict', 'triton.autotune_pointwise': False,
    })(x)
    bad = torch.nonzero(eager.view(torch.int16) != compiled.view(torch.int16)).flatten().cpu().tolist()
    result = {'torch_sha': torch.version.git_version, 'triton_distribution': importlib.metadata.version('triton'),
              'triton_file': triton.__file__, 'device': torch.cuda.get_device_name(0),
              'mismatches': len(bad), 'examples': [
                  {'input': hex(i), 'eager': hex(eager.view(torch.int16)[i].item() & 65535),
                   'compiled': hex(compiled.view(torch.int16)[i].item() & 65535)} for i in bad[:8]]}
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print('EXHAUSTIVE_STRICT_ABS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
