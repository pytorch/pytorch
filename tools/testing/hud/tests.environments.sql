-- One row per distinct way of running tests. See proposal.md section 4.3.

CREATE TABLE tests.environments
(
    -- What: identity of one way of running tests.
    -- Derived: sipHash64(<the identity fields below, in column order>) with
    --   flags sorted and values cast to the column types, computed by the
    --   ingester (section 4.1).
    -- Used: referenced as env_id by runs, health_daily and health.
    id                        UInt64,

    -- identity fields (hashed into id) ------------------------------------------

    -- What: operating system: linux, macos or windows.
    -- Derived: platform.system(), normalized.
    -- Used: "all linux"; the disable-issue tokens linux, mac and win.
    os                        LowCardinality(String),

    -- What: major.minor of the OS or container image, e.g. 22.04 or 15.6.
    -- Derived: platform.freedesktop_os_release(), platform.mac_ver(), platform.win32_ver().
    -- Used: the MACOS_VERSION gates in tests; constant per Linux image, so free there.
    os_version                LowCardinality(String),

    -- What: CPU architecture: x86_64, aarch64, s390x, ppc64le.
    -- Derived: platform.machine(), with arm64 normalized to aarch64.
    -- Used: filters; kernels and numerics differ per architecture.
    cpu_architecture          LowCardinality(String),

    -- What: CPU kernel level, override included: default, avx2, avx512, amx, sve128,
    --   sve256, vsx or zvector.
    -- Derived: torch.backends.cpu.get_cpu_capability(), lower-cased ("Z VECTOR" as
    --   zvector); avx512 becomes amx when torch.cpu._is_amx_tile_supported().
    --   Reflects the ATEN_CPU_CAPABILITY override of the nogpu_* configs.
    -- Used: which CPU kernels ran; separates AMX machines (ATen reports avx512 there
    --   too) from AVX-512 ones, and Graviton3 (sve256) from Graviton4 (sve128).
    cpu_capability            LowCardinality(String),

    -- What: interpreter major.minor with a t suffix when free-threaded, e.g. 3.10 or
    --   3.14t, as in the interpreter's own name (python3.14t).
    -- Derived: sys.version_info, plus t when sysconfig.get_config_var("Py_GIL_DISABLED")
    --   is set (sys.abiflags does not exist on Windows).
    -- Used: Python version gates; normal and free-threaded interpreters run side by
    --   side and behave differently.
    python_version            LowCardinality(String),

    -- What: compiler that built torch: gcc, clang or msvc.
    -- Derived: the CXX compiler line of torch.__config__.show().
    -- Used: "all clang builds"; concurrent gcc and clang builds of the same commit.
    cc_compiler               LowCardinality(String),

    -- What: major version of that compiler, e.g. 11 or 21; the full version goes
    --   to the report properties only.
    -- Derived: same line of torch.__config__.show().
    -- Used: separates builds that exist concurrently; patch versions never split history.
    cc_compiler_version       LowCardinality(String),

    -- What: what torch was built for: cpu, cuda, rocm, xpu, mps, tpu. A CUDA build
    --   on a CPU-only runner stays cuda with device_count = 0.
    -- Derived: torch.version.cuda / .hip / .xpu, torch.backends.mps.is_built(),
    --   TPU from the harness environment.
    -- Used: "all cuda"; the disable-issue tokens rocm and xpu.
    accelerator               LowCardinality(String),

    -- What: major.minor of the accelerator stack, e.g. 13.2; empty for cpu.
    -- Derived: the same torch.version fields.
    -- Used: "cuda 13.2 versus 13.0"; tests gate on it. Excluded from any family view.
    accelerator_version       LowCardinality(String),

    -- What: normalized accelerator model: a10g, l4, h100, b200, mi300x, mi350x, m1,
    --   m2; empty when the build cannot use a GPU. Hardware variants that share a
    --   marketing name (a partitioned MI350X and a full card) are split here.
    -- Derived: nvidia-smi, amd-smi or sysctl in a subprocess once per job, mapped
    --   through a small versioned table; never torch.cuda.get_device_name() in the
    --   test process, which would initialize CUDA before tests fork.
    -- Used: "all H100 results"; model-specific gates, memory and SM counts.
    device_name               LowCardinality(String),

    -- What: number of accelerators the test process can see, 0 for none.
    -- Derived: torch.accelerator.device_count() in the process (NVML or amdsmi, no
    --   CUDA context); honors CUDA_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES.
    -- Used: single-GPU versus multi-GPU and distributed behavior.
    device_count              UInt8,

    -- What: the harness flags set to non-default values in this process, e.g.
    --   {'PYTORCH_TEST_WITH_INDUCTOR': '1', 'PYTORCH_TEST_WITH_SLOW': '1'}.
    --   Sanitizer and debug builds show up here as PYTORCH_TEST_WITH_ASAN, _UBSAN,
    --   _TSAN and _DEBUG_BUILD.
    -- Derived: TestEnvironment.repro_env_vars, unchanged, read after common_utils is
    --   imported. Implied flags are absent by construction (an inductor process
    --   records only the inductor flag).
    -- Used: the disable-issue tokens asan, dynamo, inductor and slow; the repro
    --   command of any run; separates modes that share one CI config.
    flags                     Map(LowCardinality(String), String),

    -- bookkeeping ---------------------------------------------------------------

    -- What: when the ingester first saw the environment.
    -- Derived: ingester clock at the insert, which happens only for an unseen id.
    -- Used: first seen for the environment; version column of ReplacingMergeTree.
    first_seen_at             DateTime
)
ENGINE = ReplacingMergeTree(first_seen_at)
ORDER BY id;
