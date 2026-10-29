"""Bisect the demonstrated native MI350 XFAIL-to-XPASS interval."""

import json
from pathlib import Path
import shutil

import triton_pin_mi350_controls as controls


def main():
    controls.REPORT = controls.ROOT / 'test/test-reports/triton-pin-mi350-bisect'
    controls.WORK = controls.ROOT / 'build/triton-pin-mi350-bisect'
    controls.REPORT.mkdir(parents=True, exist_ok=True)
    controls.WORK.mkdir(parents=True, exist_ok=True)
    revisions = json.loads((Path(__file__).with_suffix('.json')).read_text())
    assert revisions[0] == '1e259627c3426df5372396ba0640b76f3deb6402'
    assert revisions[-1] == 'd364bf19639a3f4a8aecb45bca0a533e2ecb6258'
    outcomes = {len(revisions) - 1: {'log1p': 1, 'tan': 1}}
    bounds = {op: [0, len(revisions) - 1] for op in controls.NODES}

    def evaluate(index):
        sha = revisions[index]
        label = 'native-' + sha[:8]
        site = controls.build_pin(label, 'https://github.com/triton-lang/triton.git', sha)
        result = controls.test_pin(label, site, sha)
        for operation, status in result.items():
            log = (controls.REPORT / (label + '-' + operation + '.log')).read_text(errors='replace')
            if status == 0 and not ('PIN_ORIGINAL_EXPECTED_EXCEPTION' in log and 'AssertionError:' in log and 'RuntimeError:' not in log):
                raise RuntimeError('Inconclusive expected-failure control: ' + label + '/' + operation + '; inspect its traced original exception')
        outcomes[index] = result
        print('MI350_BISECT_POINT', index, sha, result, flush=True)
        (controls.REPORT / 'outcomes.json').write_text(json.dumps({
            revisions[i]: value for i, value in outcomes.items()}, indent=2))
        return result

    assert evaluate(0) == {'log1p': 0, 'tan': 0}, 'The good endpoint changed; do not continue this bisect.'
    preferred = revisions.index('b10226036e9ac8338534ff7bed63ca5bb36e047c')
    while any(high - low > 1 for low, high in bounds.values()):
        interval = max(bounds.values(), key=lambda pair: pair[1] - pair[0])
        low, high = interval
        index = preferred if low < preferred < high and preferred not in outcomes else (low + high) // 2
        result = outcomes.get(index) or evaluate(index)
        for operation, (op_low, op_high) in bounds.items():
            if op_low < index < op_high:
                bounds[operation] = [op_low, index] if result[operation] == 1 else [index, op_high]
    result = {operation: {'good': revisions[low], 'first_bad': revisions[high],
                         'good_outcome': outcomes[low][operation], 'bad_outcome': outcomes[high][operation]}
              for operation, (low, high) in bounds.items()}
    # Preserve only the final native boundary wheels for test-only verification.
    for index in {index for pair in bounds.values() for index in pair}:
        label = 'native-' + revisions[index][:8]
        debug = controls.ROOT / 'test/debug/triton-pin-mi350-bisect' / label
        debug.mkdir(parents=True, exist_ok=True)
        for wheel in (controls.WORK / label / 'source/dist').glob('*.whl'):
            shutil.copyfile(wheel, debug / wheel.name)
    (controls.REPORT / 'adjacent-results.json').write_text(json.dumps(result, indent=2))
    print('MI350_NATIVE_ADJACENT_RESULTS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
