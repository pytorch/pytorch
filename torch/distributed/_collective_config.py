# mypy: allow-untyped-defs

import operator


def _serialize_nccl_config(config):
    """Snapshot config values into graph constants, never native addresses."""
    from nccl.core import NCCLCollConfig  # pyrefly: ignore [missing-import]

    if not isinstance(config, NCCLCollConfig):
        raise TypeError("config must be an nccl.core.NCCLCollConfig")
    values = {
        "min_ctas": config.min_ctas,
        "max_ctas": config.max_ctas,
        "nvls_ctas": config.nvls_ctas,
        "cga_cluster_size": config.cga_cluster_size,
        "alg_selection": config.alg_selection,
        "force_alg_selection": config.force_alg_selection,
        "cta_policy": None if config.cta_policy is None else int(config.cta_policy),
        "user_profiler_tag": config.user_profiler_tag,
        "vendor_count": len(config.vendor_options),
    }

    for index, option in enumerate(config.vendor_options):
        if option.raw_value is not None:
            raise NotImplementedError(
                "Tracing raw-pointer vendor options requires a runtime binding"
            )
        prefix = f"vendor_{index}_"
        values[prefix + "vendor_id"] = option.vendor_id
        values[prefix + "option_id"] = option.option_id
        values[prefix + "int_value"] = option.int_value
        values[prefix + "str_value"] = option.str_value
    # Config integers are guarded constants, not dynamic tensor dimensions.
    return {
        key: operator.index(value)
        if isinstance(value, int) and not isinstance(value, bool)
        else value
        for key, value in values.items()
    }


def _deserialize_nccl_config(values):
    from nccl.core import (  # pyrefly: ignore [missing-import]
        NCCLCollConfig,
        VendorOption,
    )

    values = dict(values)
    options = []
    for index in range(values.pop("vendor_count")):
        prefix = f"vendor_{index}_"
        options.append(
            VendorOption(
                vendor_id=values.pop(prefix + "vendor_id"),
                option_id=values.pop(prefix + "option_id"),
                int_value=values.pop(prefix + "int_value"),
                str_value=values.pop(prefix + "str_value"),
            )
        )
    values["vendor_options"] = tuple(options)
    return NCCLCollConfig(**values)
