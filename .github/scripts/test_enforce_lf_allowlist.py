#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "enforce_lf_allowlist.py"


def write_arc_yaml(tmp_dir: str, lf_allowlist_yaml: str = "") -> str:
    """Write a fixture arc.yaml with the given lf_allowlist content, so
    restricted-mode tests don't touch the real arc.yaml."""
    path = Path(tmp_dir) / "arc.yaml"
    path.write_text(lf_allowlist_yaml)
    return str(path)


def run(
    matrix: str,
    github_output: str | None = None,
    lf_runners: str | None = None,
    arc_yaml: str | None = None,
) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT)]
    if lf_runners is not None:
        cmd += ["--lf-runners", lf_runners]
    cmd.append(matrix)

    env = os.environ.copy()
    if github_output is not None:
        env["GITHUB_OUTPUT"] = github_output
    else:
        env.pop("GITHUB_OUTPUT", None)
    if arc_yaml is not None:
        env["ARC_YAML_PATH"] = arc_yaml
    else:
        env.pop("ARC_YAML_PATH", None)

    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def parse_output(stdout: str) -> dict:
    """Extract the JSON matrix from the 'Setting test-matrix=...' line."""
    prefix = "Setting test-matrix="
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise ValueError(f"no test-matrix output found in: {stdout}")


def check(condition: bool, msg: str = "") -> None:
    if not condition:
        raise AssertionError(msg)


def test_empty_include_passes_through():
    matrix = """{ include: [] }"""
    result = run(matrix)
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    check(output == {"include": []}, f"expected empty include, got {output}")


def test_empty_string_passes_through():
    result = run("")
    check(result.returncode == 0, result.stderr)


def test_github_output_file():
    """When GITHUB_OUTPUT is set, the script writes test-matrix to that file."""
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "l-x86iavx512-8-64" },
    ]}"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tmp_path = f.name

    try:
        result = run(matrix, github_output=tmp_path)
        check(result.returncode == 0, result.stderr)

        contents = Path(tmp_path).read_text()
        check(
            contents.startswith("test-matrix="), f"unexpected file contents: {contents}"
        )
        written = json.loads(contents[len("test-matrix=") :].strip())
        check(written["include"][0]["runner"] == "l-x86iavx512-8-64")
    finally:
        os.unlink(tmp_path)


def test_onnx_config_excluded():
    """onnxruntime hits pthread_setaffinity_np errors on ARC k8s (huydo)."""
    matrix = """{ include: [
      { config: "onnx", shard: 1, num_shards: 1, runner: "l-x86iavx512-8-64" },
      { config: "default", shard: 1, num_shards: 1, runner: "l-x86iavx512-8-64" },
    ]}"""
    result = run(matrix)
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    configs = [e["config"] for e in output["include"]]
    check(configs == ["default"], f"expected onnx excluded, got {configs}")


def test_non_lf_entry_untouched_regardless_of_allowlist():
    """An entry not currently on 'lf-' (already 'mt-', or a ROCm/XPU-style
    passthrough label) needs no enforcement and must be left completely
    alone -- even under a restricted allowlist that would reject its bare
    pod name if it were checked."""
    with tempfile.TemporaryDirectory() as d:
        arc_yaml = write_arc_yaml(
            d,
            textwrap.dedent("""\
                lf_allowlist:
                  mode: restricted
                  runners:
                    - l-x86aavx2-11-41-a10g
                """),
        )
        matrix = """{ include: [
          { config: "default", shard: 1, num_shards: 1, runner: "mt-l-x86iamx-22-225-h100" },
          { config: "default", shard: 1, num_shards: 1, runner: "linux.rocm.gpu.2" },
        ]}"""
        result = run(matrix, arc_yaml=arc_yaml)
        check(result.returncode == 0, result.stderr)
        output = parse_output(result.stdout)
        runners = [e["runner"] for e in output["include"]]
        check(
            runners == ["mt-l-x86iamx-22-225-h100", "linux.rocm.gpu.2"],
            f"non-lf- entries must pass through unchanged, got {runners}",
        )


def test_mixed_matrix_hardcoded_mt_alongside_dynamic_lf():
    """Regression guard: a job that builds on a dynamically-resolved 'lf-'
    but hardcodes 'mt-' on specific entries (e.g. H100, as in
    inductor-periodic.yml) must not have that hardcoded entry re-prefixed
    to 'mt-mt-...' just because a sibling 'lf-' entry gets force-routed."""
    with tempfile.TemporaryDirectory() as d:
        arc_yaml = write_arc_yaml(
            d,
            textwrap.dedent("""\
                lf_allowlist:
                  mode: restricted
                  runners:
                    - l-x86aavx2-11-41-a10g
                """),
        )
        matrix = """{ include: [
          { config: "default", shard: 1, num_shards: 1, runner: "lf-l-x86iavx512-8-64" },
          { config: "default", shard: 1, num_shards: 1, runner: "mt-l-x86iamx-22-225-h100" },
        ]}"""
        result = run(matrix, arc_yaml=arc_yaml)
        check(result.returncode == 0, result.stderr)
        output = parse_output(result.stdout)
        runners = [e["runner"] for e in output["include"]]
        check(
            runners == ["mt-l-x86iavx512-8-64", "mt-l-x86iamx-22-225-h100"],
            f"unexpected double-prefix or missed enforcement: {runners}",
        )


def test_lf_runners_flag_restricts_unlisted_runner():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "lf-l-x86iavx512-16-128" },
    ]}"""
    result = run(matrix, lf_runners="l-x86aavx2-29-113-a10g")
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    check(
        output["include"][0]["runner"] == "mt-l-x86iavx512-16-128",
        f"expected mt- override, got {output['include'][0]['runner']}",
    )


def test_lf_runners_flag_allows_listed_runner():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "lf-l-x86aavx2-29-113-a10g" },
    ]}"""
    result = run(matrix, lf_runners="l-x86aavx2-29-113-a10g")
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    check(output["include"][0]["runner"] == "lf-l-x86aavx2-29-113-a10g")


def test_lf_runners_empty_string_overrides_restricted_arc_yaml():
    """An explicit empty --lf-runners means unrestricted, even if arc.yaml
    itself is restricted."""
    with tempfile.TemporaryDirectory() as d:
        arc_yaml = write_arc_yaml(
            d,
            textwrap.dedent("""\
                lf_allowlist:
                  mode: restricted
                  runners:
                    - l-x86aavx2-29-113-a10g
                """),
        )
        matrix = """{ include: [
          { config: "default", shard: 1, num_shards: 1, runner: "lf-l-x86iavx512-16-128" },
        ]}"""
        result = run(matrix, lf_runners="", arc_yaml=arc_yaml)
        check(result.returncode == 0, result.stderr)
        output = parse_output(result.stdout)
        actual = output["include"][0]["runner"]
        check(
            actual == "lf-l-x86iavx512-16-128",
            f"empty --lf-runners should mean unrestricted, got {actual}",
        )


def test_no_lf_runners_flag_falls_back_to_restricted_arc_yaml():
    with tempfile.TemporaryDirectory() as d:
        arc_yaml = write_arc_yaml(
            d,
            textwrap.dedent("""\
                lf_allowlist:
                  mode: restricted
                  runners:
                    - l-x86aavx2-29-113-a10g
                """),
        )
        matrix = """{ include: [
          { config: "default", shard: 1, num_shards: 1, runner: "lf-l-x86iavx512-16-128" },
        ]}"""
        result = run(matrix, arc_yaml=arc_yaml)
        check(result.returncode == 0, result.stderr)
        output = parse_output(result.stdout)
        actual = output["include"][0]["runner"]
        check(
            actual == "mt-l-x86iavx512-16-128",
            f"expected fallback to arc.yaml restricted mode, got {actual}",
        )


def test_no_lf_runners_flag_restricted_with_empty_runners_is_unrestricted():
    """An empty runners: list under mode: restricted must mean unrestricted,
    matching what an explicit --lf-runners "" means (ci-infra#1081) -- else
    trimming the last entry from arc.yaml silently blocks every LF job."""
    with tempfile.TemporaryDirectory() as d:
        arc_yaml = write_arc_yaml(
            d,
            textwrap.dedent("""\
                lf_allowlist:
                  mode: restricted
                  runners: []
                """),
        )
        matrix = """{ include: [
          { config: "default", shard: 1, num_shards: 1, runner: "lf-l-x86iavx512-16-128" },
        ]}"""
        result = run(matrix, arc_yaml=arc_yaml)
        check(result.returncode == 0, result.stderr)
        output = parse_output(result.stdout)
        actual = output["include"][0]["runner"]
        check(
            actual == "lf-l-x86iavx512-16-128",
            f"empty runners: under restricted mode should mean unrestricted, got {actual}",
        )


def test_no_lf_runners_flag_arc_yaml_mode_all_is_noop():
    with tempfile.TemporaryDirectory() as d:
        arc_yaml = write_arc_yaml(
            d,
            textwrap.dedent("""\
                lf_allowlist:
                  mode: all
                  runners: []
                """),
        )
        matrix = """{ include: [
          { config: "default", shard: 1, num_shards: 1, runner: "lf-l-x86iavx512-16-128" },
        ]}"""
        result = run(matrix, arc_yaml=arc_yaml)
        check(result.returncode == 0, result.stderr)
        output = parse_output(result.stdout)
        check(output["include"][0]["runner"] == "lf-l-x86iavx512-16-128")


def test_invalid_lf_allowlist_mode_fails():
    with tempfile.TemporaryDirectory() as d:
        arc_yaml = write_arc_yaml(
            d,
            textwrap.dedent("""\
                lf_allowlist:
                  mode: bogus
                  runners: []
                """),
        )
        matrix = """{ include: [
          { config: "default", shard: 1, num_shards: 1, runner: "lf-l-x86iavx512-16-128" },
        ]}"""
        result = run(matrix, arc_yaml=arc_yaml)
        check(result.returncode == 1)
        check("lf_allowlist.mode must be one of" in result.stderr, result.stderr)


def test_invalid_lf_allowlist_mode_fails_even_with_lf_runners_override():
    """Regression guard: mode validation must not be skipped just because a
    caller passes --lf-runners."""
    with tempfile.TemporaryDirectory() as d:
        arc_yaml = write_arc_yaml(
            d,
            textwrap.dedent("""\
                lf_allowlist:
                  mode: bogus
                  runners: []
                """),
        )
        matrix = """{ include: [
          { config: "default", shard: 1, num_shards: 1, runner: "lf-l-x86iavx512-16-128" },
        ]}"""
        result = run(matrix, lf_runners="l-x86iavx512-16-128", arc_yaml=arc_yaml)
        check(result.returncode == 1)
        check("lf_allowlist.mode must be one of" in result.stderr, result.stderr)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed")
