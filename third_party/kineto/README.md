# Kineto

Kineto is a library used in the PyTorch Profiler.

The Kineto project enables:
- **performance observability and diagnostics** across common ML bottleneck components
- **actionable recommendations** for common issues
- integration of external system-level profiling tools
- integration with popular visualization platforms and analysis pipelines

The central component of Kineto is Libkineto, a profiling library with special focus on low-overhead GPU timeline tracing.

## Libkineto

Libkineto is an in-process profiling library integrated with the PyTorch Profiler. Please refer to the [README](libkineto/README.md) file in the `libkineto` folder as well as documentation on the [new PyTorch Profiler API](https://pytorch.org/docs/master/profiler.html).

## Releases and Contributing
Kineto lives in the PyTorch repository and ships as part of each PyTorch release; there is no separate Kineto release.

Contributions are made as regular PyTorch pull requests, following PyTorch's [contributing guide](../../CONTRIBUTING.md). Changes under `third_party/kineto` are reviewed by the PyTorch Profiler maintainers. If you plan to contribute a new feature, please open a PyTorch issue to discuss it first.

## License
Kineto has a BSD-style license, as found in the [LICENSE](LICENSE) file.
