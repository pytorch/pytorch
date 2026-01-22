from __future__ import annotations

import copy
import logging
from functools import lru_cache
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.kernel_template_choice import KernelTemplateChoice
from torch._inductor.template_heuristics.params import DictKernelTemplateParams


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Generator

    from torch._inductor.codegen.common import KernelTemplate
    from torch._inductor.kernel_inputs import KernelInputs
    from torch._inductor.select_algorithm import ExternKernelChoice


class LookupTableChoices(InductorChoices):
    """
    InductorChoices subclass that uses lookup table when available, otherwise falls back to parent.
    All lookup functionality is contained within this class and can be customized by overriding methods.
    """

    def _get_lookup_table(self) -> dict[str, list[dict[str, Any]]]:
        """
        Get the template lookup table from config.
        Override this method to use custom lookup table sources (database, API, etc.).
        """
        if not torch.cuda.is_available() or config.lookup_table.table is None:
            return {}
        return config.lookup_table.table

    @staticmethod
    @lru_cache
    def _get_device_key(device: torch.device) -> Optional[str]:
        """
        Generate a device key for lookup table indexing.
        For CPU devices, returns None.
        For CUDA devices, returns the props.gcnArchName string.
        """
        if device.type != "cuda":
            # only cuda devices are supported, this indicates that the system is not in use
            # for this device
            return None

        # Get CUDA device properties
        props = torch.cuda.get_device_properties(device.index)
        return props.gcnArchName

    @staticmethod
    def _generate_kernel_inputs_key(kernel_inputs: KernelInputs) -> str:
        """
        Generate a key based on input node properties and scalars.
        The key includes dtype, size, and stride information for each input node,
        plus scalar values as key=value pairs separated by & signs.
        """
        # Get node information using existing methods
        dtypes = kernel_inputs.dtypes()
        shapes = kernel_inputs.shapes_hinted()
        strides = kernel_inputs.strides_hinted()

        # Create tuple of (dtype, shape_list, stride_list) for each node
        node_info = tuple(
            (dtype, list(shape), list(stride))
            for dtype, shape, stride in zip(dtypes, shapes, strides)
        )

        # Create base key from node information
        fmt_key = str(node_info)
        # Add scalar information if present
        if kernel_inputs._scalars:
            # Sort scalars for consistent key generation and join with &
            scalar_parts = [
                f"{key}={value}"
                for key, value in sorted(kernel_inputs._scalars.items())
            ]
            scalars_key = "&".join(scalar_parts)
            fmt_key = f"{fmt_key}+{scalars_key}"

        return f"{fmt_key}"

    def make_lookup_key(
        self, kernel_inputs: KernelInputs, op_name: str, include_device: bool = False
    ) -> Optional[str]:
        """
        Create a flattened lookup key from kernel inputs and operation name.
        Override this method to customize key generation.

        Args:
            kernel_inputs: KernelInputs object containing input nodes and scalars
            op_name: Operation name (e.g., "mm", "addmm")
            include_device: Whether to include device key in the generated key

        Returns:
            A string key combining device (optional), operation, and input information
        """
        device = kernel_inputs.device()
        dev_key = self._get_device_key(device)
        if dev_key is None:
            # The system does not run when dev_key is None, regardless of
            # whether include_device is True or False
            return None
        if not include_device:
            dev_key = None

        # Generate input key using our staticmethod
        input_key = self._generate_kernel_inputs_key(kernel_inputs)

        # Create the flattened lookup key
        if dev_key is not None:
            key_parts = [dev_key, input_key, op_name]
        else:
            key_parts = [input_key, op_name]

        return "+".join(key_parts)

    def make_lookup_key_variants(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Generate both device-specific and device-agnostic lookup keys.
        Override this method to customize key variant generation.

        Args:
            kernel_inputs: KernelInputs object containing input nodes and scalars
            op_name: Operation name (e.g., "mm", "addmm")

        Returns:
            Tuple of (device_key, device_agnostic_key). Either may be None if generation fails.
        """
        device_key = self.make_lookup_key(kernel_inputs, op_name, include_device=True)
        device_agnostic_key = self.make_lookup_key(
            kernel_inputs, op_name, include_device=False
        )

        return device_key, device_agnostic_key

    @staticmethod
    def _entry_is_valid(
        cfg: dict[str, Any],
        template_id: str,
        template_hash_map: Optional[dict[str, Optional[str]]],
    ) -> bool:
        """
        Check if a config entry is valid based on template hash validation.

        Args:
            cfg: Configuration dictionary that may contain a template_hash field
            template_id: The template identifier
            template_hash_map: Optional mapping from template_uid to src_hash for validation

        Returns:
            True if the config is valid and should be kept, False if it should be filtered out
        """
        # If hash checking is disabled or no hash map provided, keep the config
        if not config.lookup_table.check_src_hash or not template_hash_map:
            return True

        template_hash = template_hash_map.get(template_id)
        config_hash = cfg.get("template_hash")

        # Both hashes present - validate they match
        if template_hash is not None and config_hash is not None:
            if config_hash != template_hash:
                log.warning(
                    "Hash validation failed for template '%s': config_hash='%s' != template_hash='%s'. "
                    "Template code may have changed. Filtering out config: %s",
                    template_id,
                    config_hash,
                    template_hash,
                    {k: v for k, v in cfg.items() if k != "template_hash"},
                )
                return False
            else:
                log.debug(
                    "Hash validation passed for template '%s': hash='%s'",
                    template_id,
                    template_hash,
                )
                return True
        # Config has no hash - keep it
        elif config_hash is None:
            log.debug(
                "Config for template '%s' has no hash - keeping it (template_hash='%s')",
                template_id,
                template_hash,
            )
            return True
        # Template has no hash - keep config
        else:
            log.debug(
                "Template '%s' has no src_hash - keeping config with hash '%s'",
                template_id,
                config_hash,
            )
            return True

    def lookup_template_configs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        template_uids: list[str],
        template_hash_map: Optional[dict[str, Optional[str]]] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Unified function to look up template configurations for multiple templates.
        Override this method to customize lookup logic.

        Args:
            kernel_inputs: KernelInputs object containing input nodes and scalars
            op_name: Operation name (e.g., "mm", "addmm")
            template_uids: List of template identifiers (e.g., ["mm", "tma", "decompose_k"])
            template_hash_map: Optional mapping from template_uid to src_hash for validation

        Returns:
            {}: No lookup table in use, or no matches found for any template
            {"template_uid1": [config1, config2], ...}: Matches found, filtered configurations
        """
        lookup_table = self._get_lookup_table()
        if not lookup_table:
            log.debug("Lookup table: no table configured or CUDA unavailable")
            return {}

        # Try both key variants: device-specific first, then device-agnostic
        # If both exist, device-specific takes priority
        device_key, device_agnostic_key = self.make_lookup_key_variants(
            kernel_inputs, op_name
        )

        config_list = []

        for key_type, key in [
            ("device-specific", device_key),
            ("device-agnostic", device_agnostic_key),
        ]:
            if key is not None:
                config_list = lookup_table.get(key, [])
                if config_list:
                    log.debug(
                        "Lookup table: found %d configs using %s key '%s' for %s",
                        len(config_list),
                        key_type,
                        key,
                        op_name,
                    )
                    break
        else:
            log.debug(
                "Lookup table: no match for %s (tried keys: %s, %s) (table has %d keys)",
                op_name,
                device_key,
                device_agnostic_key,
                len(lookup_table),
            )
            return {}

        log.debug(
            "Lookup table: found %d configs for %s templates %s",
            len(config_list),
            op_name,
            template_uids,
        )
        # Group configs by template_id
        configs_by_template: dict[str, list[dict[str, Any]]] = {}
        for cfg in config_list:
            if not isinstance(cfg, dict):
                raise ValueError(
                    f"Config for {op_name} operation is not a dictionary: {cfg}"
                )
            if "template_id" not in cfg:
                raise ValueError(
                    f"Config for {op_name} operation missing required 'template_id' field: {cfg}"
                )

            template_id = cfg["template_id"]
            if template_id in template_uids:
                if template_id not in configs_by_template:
                    configs_by_template[template_id] = []
                configs_by_template[template_id].append(cfg)

        # Check template hashes and clean up template_id field
        result = {}
        for template_id, matching_configs in configs_by_template.items():
            filtered_configs = []
            for cfg in matching_configs:
                # Check template hash using helper function
                if not self._entry_is_valid(cfg, template_id, template_hash_map):
                    continue

                # Return a copy of the config, as we don't want to modify the original
                cconfig = copy.deepcopy(cfg)
                # Lastly, we have to throw out the template_id, as it's not a valid kwarg
                # and just used to identify which template the entry belongs to
                del cconfig["template_id"]
                # Similarly, the template_hash is not a valid kwarg
                cconfig.pop("template_hash", None)
                filtered_configs.append(cconfig)

            if filtered_configs:
                result[template_id] = filtered_configs

        return result

    def _finalize_template_configs(
        self,
        template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        templates: list[Union[KernelTemplate, ExternKernelChoice]],
        op_name: str,
        kwarg_overrides: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[KernelTemplateChoice]:
        """Check lookup table for hits, use those if found, otherwise fall back to parent."""
        # 1. Collect template src_hashes for validation
        template_uids = [template.uid for template in templates]
        template_hash_map = {}
        for template in templates:
            src_hash = getattr(template, "src_hash", None)
            template_hash_map[template.uid] = src_hash

        log.debug(
            "Choices: attempting lookup for %s with %d templates",
            op_name,
            len(template_uids),
        )

        # 2. Single batch lookup for all templates
        lookup_results = self.lookup_template_configs(
            kernel_inputs, op_name, template_uids, template_hash_map
        )

        # 3. Early exit if no lookup table or no matches
        if not lookup_results:  # Empty dict
            log.info("LookupChoices: lookup miss for %s, using fallback", op_name)
            return self._fallback(
                template_choices,
                kernel_inputs,
                templates,
                op_name,
                kwarg_overrides,
            )

        log.info(
            "LookupChoices: lookup hit for %s - found %d/%d templates: %s",
            op_name,
            len(lookup_results),
            len(template_uids),
            list(lookup_results.keys()),
        )

        # 4. Create KTCs only for templates with lookup entries
        return self._create_lookup_choices(
            lookup_results, templates, kernel_inputs, op_name
        )

    def _fallback(
        self,
        template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        templates: list[Union[KernelTemplate, ExternKernelChoice]],
        op_name: str,
        kwarg_overrides: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[KernelTemplateChoice]:
        """Fallback to parent if no lookup table or no matches."""
        # NOTE: this is broken out, so that subclasses are able to override this
        # to handle explicitly the situations where the lookup take had a miss vs
        # overriding the entire logic
        return super()._finalize_template_configs(
            template_choices,
            kernel_inputs,
            templates,
            op_name,
            kwarg_overrides,
        )

    def _create_lookup_choices(
        self,
        lookup_results: dict[str, list[dict[str, Any]]],
        templates: list[Union[KernelTemplate, ExternKernelChoice]],
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> list[KernelTemplateChoice]:
        """Create KernelTemplateChoice objects from lookup results using parent's get_ktc method."""
        templates_by_uid = {template.uid: template for template in templates}
        lookup_choices: list[KernelTemplateChoice] = []

        for template_uid, configs in lookup_results.items():
            template = templates_by_uid[template_uid]

            # Use parent's get_ktc method to get a generator, then get the first base KTC
            ktc_generator = self.get_ktc(kernel_inputs, template, op_name)

            try:
                base_ktc = next(ktc_generator)
            except StopIteration:
                # No configs from heuristic, skip this template
                continue

            # For each lookup config, create a KTC with the override kwargs
            for c in configs:
                lookup_ktc = KernelTemplateChoice(
                    template=base_ktc.template,
                    # use the ones from the lookup table
                    params=DictKernelTemplateParams(c),
                    extra_kwargs=base_ktc.extra_kwargs,
                    layout=base_ktc.layout,
                    inputs=base_ktc.inputs,
                )
                lookup_choices.append(lookup_ktc)

        return lookup_choices
