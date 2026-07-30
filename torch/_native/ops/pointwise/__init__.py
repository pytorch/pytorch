# CuTeDSL native pointwise (elementwise) op overrides. Importing this package
# registers the aten pointwise overrides with the _native registry.

from .overrides import register_pointwise_overrides


register_pointwise_overrides()
