"""Verify the original L4 strict BF16 abs test with both NVIDIA vectorizers."""

import hashlib
import json
import os
import sys

import triton_pin_mi350_controls as controls


NODE = 'inductor/test_strict_numerics.py::PointwiseStrictNumericsTestCUDA::test_pointwise_bitwise_abs_cuda_bfloat16'


def main():
    controls.REPORT = controls.ROOT / 'test/test-reports/triton-pin-l4-verify'
    controls.WORK = controls.ROOT / 'build/triton-pin-l4-verify'
    controls.REPORT.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, slp in [('candidate-default', None), ('candidate-slp', '1')]:
        env = os.environ.copy()
        env.pop('TRITON_USE_SLP_VECTORIZER', None)
        if slp is not None:
            env['TRITON_USE_SLP_VECTORIZER'] = slp
        env['TRITON_CACHE_DIR'] = str(controls.WORK / label / 'triton')
        env['TORCHINDUCTOR_CACHE_DIR'] = str(controls.WORK / label / 'inductor')
        env['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'
        expected_slp = slp is not None
        probe = ('import torch,triton,importlib.metadata; '
                 "assert torch.cuda.get_device_capability(0)==(8,9); "
                 "assert torch.version.cuda=='13.4'; "
                 "v=importlib.metadata.version('triton'); assert '1b9ee7ae' in v,v; "
                 f'assert triton.knobs.nvidia.use_slp_vectorizer=={expected_slp!r}; '
                 "print('L4_VERIFY_PROVENANCE',torch.version.git_version,v,triton.__file__,"
                 'triton.knobs.nvidia.use_slp_vectorizer)')
        controls.run([sys.executable, '-c', probe], cwd=controls.ROOT / 'test',
                     env=env, name=label + '-provenance', timeout=180)
        result = controls.run([sys.executable, '-m', 'pytest', '-q', '--reruns=0',
                               f'--junitxml={controls.REPORT / (label + ".xml")}', NODE],
                              cwd=controls.ROOT / 'test', env=env,
                              name=label + '-original', check=False, timeout=1200)
        log = (controls.REPORT / (label + '-original.log')).read_text(errors='replace')
        assert result == 0 and '1 passed' in log, log
        results[label] = result
    result = {
        'source_sha256': hashlib.sha256((controls.ROOT / 'test/inductor/test_strict_numerics.py').read_bytes()).hexdigest(),
        'original_node': NODE,
        'results': results,
    }
    (controls.REPORT / 'results.json').write_text(json.dumps(result, indent=2))
    print('L4_ORIGINAL_FIX_VERIFICATION', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
