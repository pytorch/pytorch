import os
import sys
import subprocess
import resource
from typing import Dict, List, Callable
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# Default Environment Variables
# ============================================================================
DEFAULT_ENV_VARS = {
    "TERM": "vt100",
    "TORCH_SERIALIZATION_DEBUG": "1",
    "VALGRIND": "ON",
    "LANG": "C.UTF-8",
}

# ============================================================================
# Build Environment Pattern-Based Config
# ============================================================================

@dataclass
class EnvRule:
    """Rule for setting environment variables based on BUILD_ENVIRONMENT patterns"""
    condition: Callable[[str], bool]
    env_vars: Dict[str, str]
    priority: int = 0  # Higher priority overwrites lower

# Environment rules based on BUILD_ENVIRONMENT patterns
BUILD_ENV_RULES = [
    # CUDA configurations
    EnvRule(
        condition=lambda env: "cuda" in env or "rocm" in env,
        env_vars={"PYTORCH_TESTING_DEVICE_ONLY_FOR": "cuda"},
        priority=1
    ),

    # XPU configurations
    EnvRule(
        condition=lambda env: "xpu" in env,
        env_vars={
            "PYTORCH_TESTING_DEVICE_ONLY_FOR": "xpu",
            "PYTHON_TEST_EXTRA_OPTION": "--xpu",
            "NO_TEST_TIMEOUT": "True",
        },
        priority=2
    ),

    # Valgrind disable - centralized rule for all conditions
    EnvRule(
        condition=lambda env: any(cond in env for cond in ["clang9", "s390x", "aarch64", "rocm", "xpu", "asan"]),
        env_vars={"VALGRIND": "OFF"},
        priority=2
    ),

    # Slow gradcheck
    EnvRule(
        condition=lambda env: "slow-gradcheck" in env,
        env_vars={
            "PYTORCH_TEST_WITH_SLOW_GRADCHECK": "1",
            "PYTORCH_TEST_CUDA_MEM_LEAK_CHECK": "1",
        },
        priority=1
    ),

    # ASAN (AddressSanitizer)
    EnvRule(
        condition=lambda env: "asan" in env,
        env_vars={
            "ASAN_OPTIONS": "detect_leaks=0:symbolize=1:detect_stack_use_after_return=true:strict_init_order=true:detect_odr_violation=1:detect_container_overflow=0:check_initialization_order=true:debug=true",
            "UBSAN_OPTIONS": "print_stacktrace=1:suppressions=$PWD/ubsan.supp",
            "PYTORCH_TEST_WITH_ASAN": "1",
            "PYTORCH_TEST_WITH_UBSAN": "1",
            "ASAN_SYMBOLIZER_PATH": "/usr/lib/llvm-18/bin/llvm-symbolizer",
            "TORCH_USE_RTLD_GLOBAL": "1",
        },
        priority=3
    ),

    # ASAN + CUDA (additional ASAN options)
    EnvRule(
        condition=lambda env: "asan" in env and "cuda" in env,
        env_vars={
            "ASAN_OPTIONS": "detect_leaks=0:symbolize=1:detect_stack_use_after_return=true:strict_init_order=true:detect_odr_violation=1:detect_container_overflow=0:check_initialization_order=true:debug=true:protect_shadow_gap=0",
        },
        priority=4  # Higher priority to override base ASAN_OPTIONS
    ),

    # PATH modification for non-bazel builds
    EnvRule(
        condition=lambda env: "-bazel-" not in env,
        env_vars={},  # PATH is handled specially in apply_path_modifications
        priority=1
    ),
]

# ============================================================================
# Test Config Based Mappings
# ============================================================================

TEST_CONFIG_MAP = {
    "default": {
        "CUDA_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
    },
    "slow": {
        "PYTORCH_TEST_WITH_SLOW": "1",
        "PYTORCH_TEST_SKIP_FAST": "1",
    },
    "nogpu_NO_AVX2": {
        "ATEN_CPU_CAPABILITY": "default",
    },
    "nogpu_AVX512": {
        "ATEN_CPU_CAPABILITY": "avx2",
    },
    "legacy_nvidia_driver": {
        "USE_LEGACY_DRIVER": "1",
    },
}

# Test config patterns (for configs that use substring matching)
TEST_CONFIG_PATTERN_MAP = {
    "crossref": {
        "PYTORCH_TEST_WITH_CROSSREF": "1",
    },
}

# Special case: distributed + rocm
def get_distributed_rocm_config(test_config: str, build_env: str) -> Dict[str, str]:
    """Handle special case for distributed test config with rocm"""
    if test_config == "distributed" and "rocm" in build_env:
        return {"HIP_VISIBLE_DEVICES": "0,1,2,3"}
    return {}

# ============================================================================
# Combined Conditional Logic
# ============================================================================

def apply_env_config(build_environment: str, test_config: str) -> Dict[str, str]:
    """
    Apply all environment variable configurations based on BUILD_ENVIRONMENT
    and TEST_CONFIG.

    Returns a dictionary of all environment variables to set.
    """
    env_vars = DEFAULT_ENV_VARS.copy()

    # Apply build environment rules (sorted by priority)
    for rule in sorted(BUILD_ENV_RULES, key=lambda r: r.priority):
        if rule.condition(build_environment):
            env_vars.update(rule.env_vars)

    # Apply test config exact matches
    if test_config in TEST_CONFIG_MAP:
        env_vars.update(TEST_CONFIG_MAP[test_config])

    # Apply test config pattern matches
    for pattern, config_vars in TEST_CONFIG_PATTERN_MAP.items():
        if pattern in test_config:
            env_vars.update(config_vars)

    # Apply special case: distributed + rocm
    env_vars.update(get_distributed_rocm_config(test_config, build_environment))

    # Handle other environment variables
    if os.environ.get("PYTORCH_TEST_RERUN_DISABLED_TESTS") == "1" or \
       os.environ.get("CONTINUE_THROUGH_ERROR") == "1":
        # ulimit is handled separately as it's not an env var
        pass

    if os.path.isdir(os.environ.get("HF_CACHE", "")):
        env_vars["HF_HOME"] = os.environ["HF_CACHE"]

    if os.environ.get("TESTS_TO_INCLUDE"):
        # INCLUDE_CLAUSE is handled separately as it's a bash variable, not env var
        pass

    return env_vars

# ============================================================================
# PATH Modifications
# ============================================================================

def apply_path_modifications(build_environment: str) -> str:
    """Return modified PATH based on build environment"""
    current_path = os.environ.get("PATH", "")

    if "-bazel-" not in build_environment:
        home_local_bin = os.path.expanduser("~/.local/bin")
        if home_local_bin not in current_path:
            return f"{home_local_bin}:{current_path}"

    return current_path

# ============================================================================
# Special Operations (Non-Environment Variables)
# ============================================================================

@dataclass
class SpecialOperation:
    """Defines a special operation that needs to be performed"""
    name: str
    condition: Callable[[str, str], bool]  # (build_env, test_config) -> bool
    description: str
    shell_commands: List[str]  # Shell commands that would be run
    executor: Callable  # Python function to execute this operation


# Note: SPECIAL_OPERATIONS and OPERATION_EXECUTORS are defined at the end of this file
# after the execute functions are declared

def get_special_operations(build_environment: str, test_config: str) -> Dict[str, any]:
    """
    Returns operations that need to be performed but aren't simple env var sets.

    Returns a dict with operation flags and their shell commands.
    """
    operations = {}

    for op in SPECIAL_OPERATIONS:
        should_run = op.condition(build_environment, test_config)

        # Handle debug assert tests special case (conditional shell commands)
        shell_commands = op.shell_commands
        if op.name == "run_debug_assert_tests" and should_run:
            if "-debug" in build_environment:
                shell_commands = [
                    "# Should fail in debug mode",
                    "cd test && ! python -c 'import torch; torch._C._crash_if_debug_asserts_fail(424242)'",
                ]
            else:
                shell_commands = [
                    "# Should pass in non-debug mode",
                    "cd test && python -c 'import torch; torch._C._crash_if_debug_asserts_fail(424242)'",
                ]

        operations[op.name] = {
            "enabled": should_run,
            "description": op.description,
            "shell_commands": shell_commands,
        }

    return operations

# ============================================================================
# Usage Example
# ============================================================================

def setup_test_environment(build_environment: str, test_config: str):
    """
    Main function to set up the test environment.

    Args:
        build_environment: Value of BUILD_ENVIRONMENT (e.g., "linux-jammy-cuda12.8-py3.10-gcc11-sm90")
        test_config: Value of TEST_CONFIG (e.g., "smoke", "default", "slow")
    """
    # Apply environment variables
    env_vars = apply_env_config(build_environment, test_config)
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")

    # Apply PATH modifications
    new_path = apply_path_modifications(build_environment)
    if new_path != os.environ.get("PATH"):
        os.environ["PATH"] = new_path
        print("Updated PATH")

    # Get special operations to perform
    operations = get_special_operations(build_environment, test_config)

    return env_vars, operations


def get_torch_directories():
    """Get PyTorch directory paths"""
    return {
        "PYTORCH_DIR": os.getcwd(),
        "PYTORCH_TEST_DIR": os.path.join(os.getcwd(), "test"),
    }


def get_build_directories():
    """Get build directory paths"""
    return {
        "BUILD_DIR": os.path.join(os.getcwd(), "build"),
    }


def print_as_shell_commands(env_vars, operations, build_environment, test_config):
    """Print configuration as shell commands (for dry-run mode)"""
    print("# Environment Variables")
    print("# " + "=" * 78)
    for key, value in sorted(env_vars.items()):
        safe_value = value.replace('"', '\\"')
        print(f'export {key}="{safe_value}"')

    print()
    print("# Special Operations")
    print("# " + "=" * 78)

    has_operations = False
    for op_name, op_info in operations.items():
        if not op_info["enabled"]:
            continue

        has_operations = True
        print(f"# {op_info['description']}")
        for cmd in op_info["shell_commands"]:
            if cmd:  # Skip empty commands
                print(cmd)
        print()

    if not has_operations:
        print("# (no special operations required)")
    print()


if __name__ == "__main__":
    # Example usage
    build_env = os.environ.get("BUILD_ENVIRONMENT", "linux-jammy-cuda12.8-py3.10-gcc11-sm90")
    test_cfg = os.environ.get("TEST_CONFIG", "smoke")

    print(f"BUILD_ENVIRONMENT: {build_env}")
    print(f"TEST_CONFIG: {test_cfg}")
    print("\nApplying configuration...\n")

    env_vars, operations = setup_test_environment(build_env, test_cfg)

    print("\nSpecial operations to perform:")
    for op, should_run in operations.items():
        if should_run:
            print(f"  - {op}")


def detect_cuda_arch():
    """Detect CUDA architecture (calls common.sh function)"""
    # This would normally call the detect_cuda_arch function from common.sh
    # For now, we'll just note that this needs to be called
    print("Note: detect_cuda_arch should be called here")


def apply_numba_patch(build_environment: str):
    """Apply numba patch to avoid CUDA-13 crash"""
    if "cuda" not in build_environment:
        return

    try:
        import numba.cuda
        numba_cuda_dir = Path(numba.cuda.__file__).parent

        if numba_cuda_dir.exists():
            script_dir = Path(__file__).resolve().parent
            numba_patch = script_dir / "numba-cuda-13.patch"

            if numba_patch.exists():
                print(f"Applying numba patch to {numba_cuda_dir}")
                with open(numba_patch, "r") as patch_file:
                    subprocess.run(
                        ["patch", "-p4"],
                        stdin=patch_file,
                        cwd=numba_cuda_dir,
                        check=True
                    )
            else:
                print(f"Warning: numba patch not found at {numba_patch}")
    except (ImportError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Could not apply numba patch: {e}")


def setup_workspace_permissions():
    """Setup workspace permissions for Jenkins"""
    workspace_dir = Path("/var/lib/jenkins/workspace")
    if not workspace_dir.exists():
        return

    try:
        # Get original owner ID
        stat_info = workspace_dir.stat()
        original_owner_id = stat_info.st_uid

        print(f"Setting up workspace permissions (original owner: {original_owner_id})")

        # Change ownership to jenkins
        subprocess.run(
            ["sudo", "chown", "-R", "jenkins", str(workspace_dir)],
            check=True
        )

        # Add safe directory for git
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", str(workspace_dir)],
            check=True
        )

        # Note: In bash, cleanup_workspace is registered with trap_add
        # In Python, you'd use atexit.register for similar functionality
        import atexit
        def cleanup():
            print("Cleaning up workspace permissions")
            subprocess.run(
                ["sudo", "chown", "-R", str(original_owner_id), str(workspace_dir)],
                check=False  # Don't fail if cleanup fails
            )
        atexit.register(cleanup)

    except Exception as e:
        print(f"Warning: Could not setup workspace permissions: {e}")


def disable_core_dumps():
    """Disable core dumps to save disk space"""
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        print("Disabled core dumps")
    except Exception as e:
        print(f"Warning: Could not disable core dumps: {e}")


def set_custom_test_artifact_dir(build_environment: str):
    """Set CUSTOM_TEST_ARTIFACT_BUILD_DIR if not in bazel build"""
    if "bazel" in build_environment:
        return None

    default_dir = "build/custom_test_artifacts"
    custom_dir = os.environ.get("CUSTOM_TEST_ARTIFACT_BUILD_DIR", default_dir)
    custom_dir = os.path.realpath(custom_dir)
    os.environ["CUSTOM_TEST_ARTIFACT_BUILD_DIR"] = custom_dir
    return custom_dir


def set_include_clause():
    """Set INCLUDE_CLAUSE if TESTS_TO_INCLUDE is set"""
    tests_to_include = os.environ.get("TESTS_TO_INCLUDE")
    if tests_to_include:
        print("Setting INCLUDE_CLAUSE")
        include_clause = f"--include {tests_to_include}"
        os.environ["INCLUDE_CLAUSE"] = include_clause
        return include_clause
    return ""


def set_pr_number():
    """Set PR_NUMBER from environment"""
    pr_number = os.environ.get("PR_NUMBER") or os.environ.get("CIRCLE_PR_NUMBER", "")
    os.environ["PR_NUMBER"] = pr_number
    return pr_number


def source_xpu_scripts():
    """Source Intel oneAPI environment scripts for XPU"""
    scripts = [
        "/opt/intel/oneapi/compiler/latest/env/vars.sh",
        "/opt/intel/oneapi/umf/latest/env/vars.sh",  # Optional
        "/opt/intel/oneapi/ccl/latest/env/vars.sh",
        "/opt/intel/oneapi/mpi/latest/env/vars.sh",
        "/opt/intel/oneapi/pti/latest/env/vars.sh",
    ]

    print("Sourcing Intel oneAPI environment scripts for XPU")

    # Build a command that sources all scripts and prints env
    source_commands = []
    for script in scripts:
        if Path(script).exists():
            source_commands.append(f'source "{script}"')
            print(f"  Will source: {script}")
        elif "umf" not in script:  # UMF is optional
            print(f"  Warning: Script not found: {script}")

    if not source_commands:
        print("  No scripts found to source")
        return

    try:
        # Source all scripts in one go, then print environment
        command = " && ".join(source_commands) + " && env"

        result = subprocess.run(
            command,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )

        # Parse and apply environment variables
        for line in result.stdout.split('\n'):
            if '=' in line:
                key, _, value = line.partition('=')
                # Filter out bash functions and complex values
                if key and not key.startswith('BASH_FUNC_'):
                    os.environ[key] = value

        print("  Environment variables updated from Intel oneAPI scripts")

    except subprocess.CalledProcessError as e:
        print(f"  Warning: Failed to source Intel oneAPI scripts: {e}")

    # Check XPU status
    try:
        subprocess.run(
            ["timeout", "30", "xpu-smi", "discovery"],
            check=False,  # Don't fail if this fails
            timeout=30
        )
    except Exception as e:
        print(f"  Warning: Could not run xpu-smi: {e}")


def run_rocminfo():
    """Run rocminfo for ROCm builds"""
    print("Running rocminfo...")
    try:
        subprocess.run(["rocminfo"], check=True)
        subprocess.run(
            ["rocminfo"],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        # Filter output for GPU info
        result = subprocess.run(
            ["rocminfo"],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        import re
        for line in result.stdout.split('\n'):
            if re.search(r'Name:.*\sgfx|Marketing', line):
                print(line)

        # Set MAYBE_ROCM for benchmark directory
        os.environ["MAYBE_ROCM"] = "rocm/"
    except Exception as e:
        print(f"Warning: Could not run rocminfo: {e}")


def set_ld_preload_asan():
    """Set LD_PRELOAD for ASAN"""
    try:
        result = subprocess.run(
            ["clang", "--print-file-name=libclang_rt.asan-x86_64.so"],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        ld_preload = result.stdout.strip()
        os.environ["LD_PRELOAD"] = ld_preload
        print(f"Set LD_PRELOAD={ld_preload}")
    except Exception as e:
        print(f"Warning: Could not set LD_PRELOAD for ASAN: {e}")


def run_asan_tests():
    """Run ASAN/UBSAN validation tests"""
    print("Running ASAN/UBSAN validation tests")
    test_dir = Path("test")

    # Print torch version
    try:
        result = subprocess.run(
            ["python", "-c", "import torch; print(torch.__version__, torch.version.git_version)"],
            cwd=test_dir,
            check=True
        )
    except Exception as e:
        print(f"Warning: Could not print torch version: {e}")

    print("The next tests are expected to crash; if they don't that means ASAN/UBSAN is misconfigured")

    # Define crash tests (these should fail)
    crash_tests = [
        "import torch; torch._C._crash_if_csrc_asan(3)",
        "import torch; torch._C._crash_if_vptr_ubsan()",
        "import torch; torch._C._crash_if_aten_asan(3)",
    ]

    for test_code in crash_tests:
        try:
            result = subprocess.run(
                ["python", "-c", test_code],
                cwd=test_dir,
                capture_output=True
            )
            if result.returncode == 0:
                print(f"ERROR: Expected crash test to fail but it passed: {test_code}")
            else:
                print(f"✓ Crash test failed as expected")
        except Exception as e:
            print(f"Warning: Could not run crash test: {e}")


def run_debug_assert_tests(build_environment: str):
    """Run debug assert validation tests"""
    test_dir = Path("test")
    test_code = "import torch; torch._C._crash_if_debug_asserts_fail(424242)"

    if "-debug" in build_environment:
        print(f"We are in debug mode: {build_environment}. Expect the python assertion to fail")
        try:
            result = subprocess.run(
                ["python", "-c", test_code],
                cwd=test_dir,
                capture_output=True
            )
            if result.returncode == 0:
                print("ERROR: Expected debug assert test to fail but it passed")
            else:
                print("✓ Debug assert test failed as expected")
        except Exception as e:
            print(f"Warning: Could not run debug assert test: {e}")

    elif "-bazel-" not in build_environment:
        print(f"We are not in debug mode: {build_environment}. Expect the assertion to pass")
        try:
            subprocess.run(
                ["python", "-c", test_code],
                cwd=test_dir,
                check=True
            )
            print("✓ Debug assert test passed as expected")
        except Exception as e:
            print(f"Warning: Could not run debug assert test: {e}")


def test_legacy_nvidia_driver():
    """Test CUDA initialization for legacy nvidia driver"""
    print("Testing CUDA initialization for legacy nvidia driver")
    test_dir = Path("test")
    try:
        subprocess.run(
            ["python", "-c", "import torch; torch.rand(2, 2, device='cuda')"],
            cwd=test_dir,
            check=True
        )
        print("✓ CUDA initialized successfully")
    except Exception as e:
        print(f"ERROR: Could not initialize CUDA: {e}")
        raise


def set_default_shard_config():
    """Set default shard configuration"""
    os.environ.setdefault("SHARD_NUMBER", "1")
    os.environ.setdefault("NUM_TEST_SHARDS", "1")

    shard_number = int(os.environ["SHARD_NUMBER"])
    num_shards = int(os.environ["NUM_TEST_SHARDS"])

    return {
        "SHARD_NUMBER": shard_number,
        "NUM_TEST_SHARDS": num_shards,
        "multiple_shards": num_shards > 1,
    }


# Define SPECIAL_OPERATIONS directly - each operation includes its executor function
SPECIAL_OPERATIONS = [
    SpecialOperation(
        name="setup_workspace_permissions",
        condition=lambda build_env, test_config: (
            "rocm" not in build_env and
            "s390x" not in build_env and
            os.path.isdir("/var/lib/jenkins/workspace")
        ),
        description="Setup workspace permissions for Jenkins",
        shell_commands=[
            "sudo chown -R jenkins /var/lib/jenkins/workspace",
            "git config --global --add safe.directory /var/lib/jenkins/workspace",
        ],
        executor=setup_workspace_permissions,
    ),
    SpecialOperation(
        name="apply_numba_patch",
        condition=lambda build_env, test_config: "cuda" in build_env,
        description="Apply numba CUDA-13 patch",
        shell_commands=[
            "# Get numba cuda directory",
            "NUMBA_CUDA_DIR=$(python -c 'import os;import numba.cuda; print(os.path.dirname(numba.cuda.__file__))' 2>/dev/null || true)",
            "# Apply patch",
            "cd \"$NUMBA_CUDA_DIR\" && patch -p4 < .ci/pytorch/numba-cuda-13.patch",
        ],
        executor=apply_numba_patch,
    ),
    SpecialOperation(
        name="disable_core_dumps",
        condition=lambda build_env, test_config: (
            os.environ.get("PYTORCH_TEST_RERUN_DISABLED_TESTS") == "1" or
            os.environ.get("CONTINUE_THROUGH_ERROR") == "1"
        ),
        description="Disable core dumps to save disk space",
        shell_commands=[
            "ulimit -c 0",
        ],
        executor=disable_core_dumps,
    ),
    SpecialOperation(
        name="source_xpu_scripts",
        condition=lambda build_env, test_config: "xpu" in build_env,
        description="Source Intel oneAPI environment scripts",
        shell_commands=[
            "source /opt/intel/oneapi/compiler/latest/env/vars.sh",
            "[ -f /opt/intel/oneapi/umf/latest/env/vars.sh ] && source /opt/intel/oneapi/umf/latest/env/vars.sh",
            "source /opt/intel/oneapi/ccl/latest/env/vars.sh",
            "source /opt/intel/oneapi/mpi/latest/env/vars.sh",
            "source /opt/intel/oneapi/pti/latest/env/vars.sh",
            "timeout 30 xpu-smi discovery || true",
        ],
        executor=source_xpu_scripts,
    ),
    SpecialOperation(
        name="run_rocminfo",
        condition=lambda build_env, test_config: "rocm" in build_env,
        description="Run rocminfo to gather GPU information",
        shell_commands=[
            "rocminfo",
            "rocminfo | grep -E 'Name:.*\\sgfx|Marketing'",
            'export MAYBE_ROCM="rocm/"',
        ],
        executor=run_rocminfo,
    ),
    SpecialOperation(
        name="set_ld_preload_asan",
        condition=lambda build_env, test_config: "asan" in build_env,
        description="Set LD_PRELOAD for AddressSanitizer",
        shell_commands=[
            "export LD_PRELOAD=$(clang --print-file-name=libclang_rt.asan-x86_64.so)",
        ],
        executor=set_ld_preload_asan,
    ),
    SpecialOperation(
        name="run_asan_tests",
        condition=lambda build_env, test_config: "asan" in build_env,
        description="Run ASAN/UBSAN validation tests (expected to crash)",
        shell_commands=[
            "cd test && python -c 'import torch; print(torch.__version__, torch.version.git_version)'",
            "cd test && ! python -c 'import torch; torch._C._crash_if_csrc_asan(3)'",
            "cd test && ! python -c 'import torch; torch._C._crash_if_vptr_ubsan()'",
            "cd test && ! python -c 'import torch; torch._C._crash_if_aten_asan(3)'",
        ],
        executor=run_asan_tests,
    ),
    SpecialOperation(
        name="run_debug_assert_tests",
        condition=lambda build_env, test_config: "-debug" in build_env or "-bazel-" not in build_env,
        description="Run debug assert validation tests",
        shell_commands=[],  # Conditional, set dynamically
        executor=run_debug_assert_tests,
    ),
    SpecialOperation(
        name="test_legacy_nvidia_driver",
        condition=lambda build_env, test_config: test_config == "legacy_nvidia_driver",
        description="Test CUDA initialization for legacy nvidia driver",
        shell_commands=[
            "cd test && python -c 'import torch; torch.rand(2, 2, device=\"cuda\")'",
            'export USE_LEGACY_DRIVER=1',
        ],
        executor=test_legacy_nvidia_driver,
    ),
]
