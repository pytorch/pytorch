# Perfetto C++ Tracing SDK (vendored amalgamation)

This directory vendors the Perfetto C++ Tracing SDK *amalgamation* -- the two
self-contained files `sdk/perfetto.h` and `sdk/perfetto.cc`. The CUPTI monitor's
native `.pftrace` encoder (`torch/csrc/profiler/cupti/monitor_pftrace.cpp`) uses
the protozero message builders from these files to emit Perfetto traces.

Only the amalgamation is vendored, not the full Perfetto source tree: the
amalgamation is designed to be self-contained (requires only a C++17 standard
library), and as of v48 Perfetto no longer checks the amalgamation into its git
tree, so a submodule would pull ~100k files the build never compiles.

## Version

- Perfetto **v56.1** (`PERFETTO_VERSION_STRING() == "v56.1-c794fceab"`,
  upstream commit `c794fceabe584dc9172e5512aaaeecc21019a635`).

## Updating

Pick the upstream version, then regenerate the two files (a maintainer step on a
networked machine -- not part of the build):

- Download `perfetto-cpp-sdk-src.zip` from
  https://github.com/google/perfetto/releases and extract `perfetto.h` /
  `perfetto.cc` into `sdk/`, **or**
- From a Perfetto checkout at the chosen tag:
  `tools/gen_amalgamated --output sdk/perfetto`.

Then refresh `LICENSE` from the same tag and update the version above.
