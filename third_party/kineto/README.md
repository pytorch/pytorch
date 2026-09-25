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
We will follow the PyTorch release schedule which roughly happens on a 3 month basis.

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the infrastructure in a different direction than you might be aware of. We expect the architecture to keep evolving.

## License
Kineto has a BSD-style license, as found in the [LICENSE](LICENSE) file.
