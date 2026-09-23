#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "map_ec2_to_arc.py"

# Minimal runner_mapping shared by the lf_allowlist fixture tests below.
FIXTURE_RUNNER_MAPPING = textwrap.dedent("""\
    runner_mapping:
      linux.4xlarge: l-x86iavx512-16-128
      linux.g5.4xlarge.nvidia.gpu: l-x86aavx2-29-113-a10g
    meta_only_runners: []
    """)


def write_arc_yaml(tmp_dir: str, lf_allowlist_yaml: str = "") -> str:
    """Write a fixture arc.yaml with FIXTURE_RUNNER_MAPPING plus optional
    lf_allowlist content, so restricted-mode tests don't touch the real
    arc.yaml."""
    path = Path(tmp_dir) / "arc.yaml"
    path.write_text(FIXTURE_RUNNER_MAPPING + lf_allowlist_yaml)
    return str(path)


def run(
    matrix: str,
    prefix: str = "",
    github_output: str | None = None,
    lf_runners: str | None = None,
    arc_yaml: str | None = None,
) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT)]
    if prefix:
        cmd += ["--prefix", prefix]
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


def test_basic_matrix():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "linux.4xlarge" },
      { config: "openreg", shard: 1, num_shards: 1, runner: "linux.2xlarge" },
    ]}"""
    result = run(matrix)
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    runners = [e["runner"] for e in output["include"]]
    check(runners == ["l-x86iavx512-16-128", "l-x86iavx512-8-64"])


def test_matrix_with_prefix():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 7, runner: "mt-linux.4xlarge" },
      { config: "default", shard: 2, num_shards: 7, runner: "mt-linux.4xlarge" },
      { config: "openreg", shard: 1, num_shards: 1, runner: "mt-linux.2xlarge" },
    ]}"""
    result = run(matrix, prefix="mt-")
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    runners = [e["runner"] for e in output["include"]]
    check(
        runners
        == [
            "mt-l-x86iavx512-16-128",
            "mt-l-x86iavx512-16-128",
            "mt-l-x86iavx512-8-64",
        ]
    )


def test_matrix_without_prefix_when_none_present():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "linux.g5.4xlarge.nvidia.gpu" },
    ]}"""
    result = run(matrix)
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    check(output["include"][0]["runner"] == "l-x86aavx2-29-113-a10g")


def test_unknown_runner_fails():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "bogus.runner" },
    ]}"""
    result = run(matrix)
    check(result.returncode == 1)
    check("no ARC runner found for 'bogus.runner'" in result.stderr)


def test_prefix_not_present_on_runner():
    """When --prefix is given but a runner doesn't have it, the raw label is used."""
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "linux.4xlarge" },
    ]}"""
    result = run(matrix, prefix="mt-")
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    check(output["include"][0]["runner"] == "mt-l-x86iavx512-16-128")


def test_preserves_non_runner_fields():
    matrix = """{ include: [
      { config: "default", shard: 3, num_shards: 7, runner: "linux.large" },
    ]}"""
    result = run(matrix)
    check(result.returncode == 0, result.stderr)
    entry = parse_output(result.stdout)["include"][0]
    check(entry["config"] == "default")
    check(entry["shard"] == 3)
    check(entry["num_shards"] == 7)
    check(entry["runner"] == "l-x86iavx512-2-4")


def test_empty_include_passes_through():
    matrix = """{ include: [] }"""
    result = run(matrix)
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    check(output == {"include": []}, f"expected empty include, got {output}")


def test_empty_string_passes_through():
    result = run("")
    check(result.returncode == 0, result.stderr)


def test_mixed_runners():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "linux.4xlarge" },
      { config: "gpu", shard: 1, num_shards: 1, runner: "linux.g5.4xlarge.nvidia.gpu" },
      { config: "arm", shard: 1, num_shards: 1, runner: "linux.arm64.2xlarge" },
    ]}"""
    result = run(matrix)
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    runners = [e["runner"] for e in output["include"]]
    check(
        runners
        == [
            "l-x86iavx512-16-128",
            "l-x86aavx2-29-113-a10g",
            "l-arm64g2-6-32",
        ]
    )


def test_passthrough_runner_no_prefix():
    """Passthrough runners (ROCm, XPU) should not get the OSDC prefix."""
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 3, runner: "linux.rocm.gpu.2" },
      { config: "default", shard: 1, num_shards: 1, runner: "linux.idc.xpu" },
    ]}"""
    result = run(matrix, prefix="mt-")
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    runners = [e["runner"] for e in output["include"]]
    check(
        runners == ["linux.rocm.gpu.2", "linux.idc.xpu"],
        f"passthrough runners should not get prefix, got {runners}",
    )


def test_github_output_file():
    """When GITHUB_OUTPUT is set, the script writes test-matrix to that file."""
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "linux.2xlarge" },
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


def test_h100_forces_mt_prefix_when_no_prefix():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "linux.aws.h100" },
    ]}"""
    result = run(matrix)
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    check(output["include"][0]["runner"] == "mt-l-x86iamx-22-225-h100")


def test_h100_forces_mt_prefix_overriding_lf():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "lf-linux.aws.h100" },
    ]}"""
    result = run(matrix, prefix="lf-")
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    check(
        output["include"][0]["runner"] == "mt-l-x86iamx-22-225-h100",
        f"expected mt- override, got {output['include'][0]['runner']}",
    )


def test_h100_with_mt_prefix_idempotent():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "mt-linux.aws.h100" },
    ]}"""
    result = run(matrix, prefix="mt-")
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    check(output["include"][0]["runner"] == "mt-l-x86iamx-22-225-h100")


def test_b200_passthrough_regardless_of_prefix():
    """B200 stays on the EC2 runners until OSDC has capacity."""
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "lf-linux.dgx.b200" },
      { config: "default", shard: 1, num_shards: 1, runner: "lf-linux.dgx.b200.8" },
    ]}"""
    result = run(matrix, prefix="lf-")
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    runners = [e["runner"] for e in output["include"]]
    check(
        runners == ["linux.dgx.b200", "linux.dgx.b200.8"],
        f"b200 should pass through unprefixed, got {runners}",
    )


def test_non_h100_keeps_lf_prefix():
    """Regression guard: non-H100/B200 runners must not be forced to mt-."""
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "lf-linux.4xlarge" },
    ]}"""
    result = run(matrix, prefix="lf-")
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    check(output["include"][0]["runner"] == "lf-l-x86iavx512-16-128")


def test_h100_multi_gpu_variants_force_mt():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "lf-linux.aws.h100.4" },
      { config: "default", shard: 1, num_shards: 1, runner: "lf-linux.aws.h100.8" },
    ]}"""
    result = run(matrix, prefix="lf-")
    check(result.returncode == 0, result.stderr)
    output = parse_output(result.stdout)
    runners = [e["runner"] for e in output["include"]]
    check(
        runners
        == [
            "mt-l-x86iamx-88-900-h100-4",
            "mt-l-bx86iamx-176-1800-h100-8",
        ],
        f"unexpected runners: {runners}",
    )


def test_lf_runners_flag_restricts_unlisted_runner():
    with tempfile.TemporaryDirectory() as d:
        arc_yaml = write_arc_yaml(d)
        matrix = """{ include: [
          { config: "default", shard: 1, num_shards: 1, runner: "lf-linux.4xlarge" },
        ]}"""
        result = run(
            matrix, prefix="lf-", lf_runners="l-x86aavx2-29-113-a10g", arc_yaml=arc_yaml
        )
        check(result.returncode == 0, result.stderr)
        output = parse_output(result.stdout)
        check(
            output["include"][0]["runner"] == "mt-l-x86iavx512-16-128",
            f"expected mt- override, got {output['include'][0]['runner']}",
        )


def test_lf_runners_flag_allows_listed_runner():
    with tempfile.TemporaryDirectory() as d:
        arc_yaml = write_arc_yaml(d)
        gpu_runner = "linux.g5.4xlarge.nvidia.gpu"
        matrix = f"""{{ include: [
          {{ config: "default", shard: 1, num_shards: 1, runner: "lf-{gpu_runner}" }},
        ]}}"""
        result = run(
            matrix, prefix="lf-", lf_runners="l-x86aavx2-29-113-a10g", arc_yaml=arc_yaml
        )
        check(result.returncode == 0, result.stderr)
        output = parse_output(result.stdout)
        check(output["include"][0]["runner"] == "lf-l-x86aavx2-29-113-a10g")


def test_lf_runners_flag_ignored_when_prefix_is_mt():
    """--lf-runners only matters for lf- prefixed requests."""
    with tempfile.TemporaryDirectory() as d:
        arc_yaml = write_arc_yaml(d)
        matrix = """{ include: [
          { config: "default", shard: 1, num_shards: 1, runner: "mt-linux.4xlarge" },
        ]}"""
        result = run(
            matrix, prefix="mt-", lf_runners="l-x86aavx2-29-113-a10g", arc_yaml=arc_yaml
        )
        check(result.returncode == 0, result.stderr)
        output = parse_output(result.stdout)
        check(output["include"][0]["runner"] == "mt-l-x86iavx512-16-128")


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
          { config: "default", shard: 1, num_shards: 1, runner: "lf-linux.4xlarge" },
        ]}"""
        result = run(matrix, prefix="lf-", lf_runners="", arc_yaml=arc_yaml)
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
          { config: "default", shard: 1, num_shards: 1, runner: "lf-linux.4xlarge" },
        ]}"""
        result = run(matrix, prefix="lf-", arc_yaml=arc_yaml)
        check(result.returncode == 0, result.stderr)
        output = parse_output(result.stdout)
        actual = output["include"][0]["runner"]
        check(
            actual == "mt-l-x86iavx512-16-128",
            f"expected fallback to arc.yaml restricted mode, got {actual}",
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
          { config: "default", shard: 1, num_shards: 1, runner: "lf-linux.4xlarge" },
        ]}"""
        result = run(matrix, prefix="lf-", arc_yaml=arc_yaml)
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
          { config: "default", shard: 1, num_shards: 1, runner: "lf-linux.4xlarge" },
        ]}"""
        result = run(matrix, prefix="lf-", arc_yaml=arc_yaml)
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
          { config: "default", shard: 1, num_shards: 1, runner: "lf-linux.4xlarge" },
        ]}"""
        result = run(
            matrix,
            prefix="lf-",
            lf_runners="l-x86iavx512-16-128",
            arc_yaml=arc_yaml,
        )
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
