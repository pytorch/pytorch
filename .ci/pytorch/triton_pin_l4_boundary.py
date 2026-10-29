"""Full native adjacent controls for the NVIDIA vectorizer change."""

import importlib.metadata
import json
import os
import sys

import triton_pin_mi350_controls as controls


NODE = 'inductor/test_strict_numerics.py::PointwiseStrictNumericsTestCUDA::test_pointwise_bitwise_abs_cuda_bfloat16'


def test(label, sha, site=None, slp=None):
    env = os.environ.copy()
    if site:
        env['PYTHONPATH'] = str(site) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env['PIN_PROBE_SITE'] = str(site)
    env['PIN_PROBE_SHA'] = sha
    env['TRITON_CACHE_DIR'] = str(controls.WORK / label / 'triton')
    env['TORCHINDUCTOR_CACHE_DIR'] = str(controls.WORK / label / 'inductor')
    env['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'
    env.pop('TRITON_USE_SLP_VECTORIZER', None)
    if slp is not None:
        env['TRITON_USE_SLP_VECTORIZER'] = str(slp)
    debug = controls.ROOT / 'test/debug/triton-pin-l4-boundary' / label
    controls.run([sys.executable, str(controls.ROOT / '.ci/pytorch/triton_pin_l4_codegen_probe.py'), str(debug)],
                 cwd=controls.ROOT / 'test', env=env, name=label + '-all-encodings', timeout=1200)
    result = controls.run([sys.executable, '-m', 'pytest', '-s', '--tb=short', '--reruns=0',
                          f'--junitxml={controls.REPORT / (label + ".xml")}', NODE],
                         cwd=controls.ROOT / 'test', env=env, name=label + '-original', check=False, timeout=1200)
    log = (controls.REPORT / (label + '-original.log')).read_text(errors='replace')
    assert result in (0, 1) and ('1 passed' in log or 'forward now matches eager under strict numerics' in log)
    return result


def main():
    controls.REPORT = controls.ROOT / 'test/test-reports/triton-pin-l4-boundary'
    controls.WORK = controls.ROOT / 'build/triton-pin-l4-boundary'
    controls.REPORT.mkdir(parents=True, exist_ok=True)
    controls.WORK.mkdir(parents=True, exist_ok=True)
    candidate = '1b9ee7ae35e6e56e3f1e53650154910bdbeb4768'
    assert candidate[:8] in importlib.metadata.version('triton')
    # Restore the earlier vectorizer only as an independent mechanism control.
    results = {'candidate-default': test('candidate-default', candidate),
               'candidate-old-vectorizer': test('candidate-old-vectorizer', candidate, slp=1)}
    for label, sha in [
        ('before-vectorizer', '850348bef5a2623ea4bfeac13692ad4ae53295bd'),
        ('after-vectorizer', '78e80b9e9f7bf720945f828947e6e196284f7e6a'),
    ]:
        site = controls.build_pin(label, 'https://github.com/triton-lang/triton.git', sha)
        results[label] = test(label, sha, site)
        (controls.REPORT / 'results.json').write_text(json.dumps(results, indent=2))
    print('L4_NATIVE_VECTORIZER_BOUNDARY', json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
