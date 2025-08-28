"""
Core orchestration logic for performance model interface.

This module provides the main entry point for performance model prediction
and choice filtering in the inductor kernel selection process.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

from .registry import get_functions_for_templates

if TYPE_CHECKING:
    from ..kernel_template_choice import KernelTemplateChoice


log = logging.getLogger(__name__)


def filter_and_sort_choices(
    ktc_stack: list[KernelTemplateChoice],
    topk: int,
    discard_unranked: bool = False,
) -> list[KernelTemplateChoice]:
    """
    Filter and sort KernelTemplateChoice objects based on performance predictions.

    Args:
        ktc_stack: List of KernelTemplateChoice objects
        topk: Number of top choices to keep (-1 for all ranked choices)
        discard_unranked: Whether to discard choices without performance predictions

    Returns:
        Filtered and sorted list of KernelTemplateChoice objects
    """
    # Separate choices with predictions from those without
    ranked_choices = []
    unranked_choices = []

    for ktc in ktc_stack:
        if ktc.performance_prediction is not None:
            ranked_choices.append(ktc)
        else:
            unranked_choices.append(ktc)

    log.debug(
        f"Found {len(ranked_choices)} ranked choices and {len(unranked_choices)} unranked choices"
    )

    # Sort ranked choices by performance prediction (lower is better - shorter runtime)
    ranked_choices.sort(key=lambda x: x.performance_prediction)

    # Apply topk filtering
    filtered_choices = []

    if topk == -1:
        # Return all ranked choices plus unranked if discard_unranked=False
        filtered_choices = ranked_choices.copy()
    else:
        # Return top-k ranked choices
        filtered_choices = ranked_choices[:topk]

    # Add unranked choices based on discard_unranked flag
    if not discard_unranked:
        filtered_choices.extend(unranked_choices)
        log.debug(
            f"Keeping {len(unranked_choices)} unranked choices (discard_unranked=False)"
        )
    else:
        log.debug(
            f"Discarding {len(unranked_choices)} unranked choices (discard_unranked=True)"
        )

    log.info(
        f"Performance model filtering: "
        f"original={len(ktc_stack)}, ranked={len(ranked_choices)}, "
        f"filtered={len(filtered_choices)}, topk={topk}, discard_unranked={discard_unranked}"
    )

    return filtered_choices


def predict_and_filter_choices(
    ktc_stack: list[KernelTemplateChoice],
    template_uid_to_ktc: dict[str, list[KernelTemplateChoice]],
    op_name: str,
    topk: int = 0,
    discard_unranked: bool = False,
) -> list[KernelTemplateChoice]:
    """
    Use performance models to predict performance and filter kernel template choices.

    This is the main entry point for the performance model interface. It takes
    a list of KernelTemplateChoice objects and a mapping of template UIDs to KTCs,
    uses registered performance models to predict performance, and returns the
    filtered choices.

    Args:
        ktc_stack: List of KernelTemplateChoice objects from fallback selection
        template_uid_to_ktc: Mapping from template UID strings to lists of KTCs
        op_name: Operation name (e.g., "mm", "bmm", "addmm")

    Returns:
        List of KernelTemplateChoice objects with performance predictions populated.
        May be filtered/reordered based on performance model predictions.
    """
    if not ktc_stack:
        log.debug("Empty KTC stack provided, returning empty list")
        return []

    if not template_uid_to_ktc:
        log.debug(
            "Empty template_uid_to_ktc mapping provided, returning original stack"
        )
        return ktc_stack

    # If topk is 0, performance model is disabled - return original stack
    if topk == 0:
        log.debug("Performance model disabled (topk=0), returning original stack")
        return ktc_stack

    # Assert that topk is valid
    assert topk > 0 or topk == -1, f"topk must be > 0 or -1, got {topk}"

    # Assert that all KTCs have the same device name
    device_names = {ktc.inputs.device_name for ktc in ktc_stack}
    assert (
        len(device_names) == 1
    ), f"All KTCs must have the same device_name, but found: {device_names}"
    hardware_name = next(iter(device_names))

    assert hardware_name, "device_name must be non-empty"

    # Get all template UIDs that have KTCs
    template_uids = list(template_uid_to_ktc.keys())
    log.debug(
        f"Processing {len(ktc_stack)} KTCs across {len(template_uids)} template UIDs: {template_uids} "
        f"for device '{hardware_name}'"
    )

    # Get functions that can handle these template UIDs
    functions_to_templates = get_functions_for_templates(
        template_uids, op_name, hardware_name
    )

    if not functions_to_templates:
        log.debug(
            f"No performance model functions found for templates {template_uids} "
            f"with op_name='{op_name}', hardware_name='{hardware_name}'. "
            "Returning original KTC stack without predictions."
        )
        return ktc_stack

    log.info(
        f"Found {len(functions_to_templates)} performance model functions for "
        f"op_name='{op_name}', hardware_name='{hardware_name}'"
    )

    # Step 1: Send KTCs to the right functions and call them for predictions
    ktcs_processed = set()

    for func, template_uids_for_func in functions_to_templates.items():
        # Collect all KTCs that this function should handle
        ktcs_for_func = []
        for template_uid in template_uids_for_func:
            if template_uid in template_uid_to_ktc:
                ktcs_for_template = template_uid_to_ktc[template_uid]
                ktcs_for_func.extend(ktcs_for_template)
                ktcs_processed.update(ktcs_for_template)

        if ktcs_for_func:
            log.debug(
                f"Calling function "
                f"with {len(ktcs_for_func)} KTCs for templates {template_uids_for_func}"
            )

            try:
                # Call the function - this should populate performance_prediction field
                predicted_ktcs = func(ktcs_for_func, op_name)

                # Verify the function returned the same KTCs (as per interface contract)
                if predicted_ktcs is not ktcs_for_func:
                    log.warning(
                        "Performance model function "
                        "returned different KTC objects than provided. "
                        "Functions should modify KTCs in-place and return the same list."
                    )

                # Log successful predictions
                successful_predictions = sum(
                    1 for ktc in ktcs_for_func if ktc.performance_prediction is not None
                )
                log.debug(
                    f"Function made predictions for "
                    f"{successful_predictions}/{len(ktcs_for_func)} KTCs"
                )

            except Exception as e:
                log.error(f"Performance model function failed during prediction: {e}")
                # Continue with other functions even if one fails
                continue

    # Log KTCs that weren't processed by any model
    unprocessed_ktcs = []
    for ktc in ktc_stack:
        if ktc not in ktcs_processed:
            unprocessed_ktcs.append(ktc)

    if unprocessed_ktcs:
        log.debug(
            f"{len(unprocessed_ktcs)}/{len(ktc_stack)} KTCs were not processed "
            "by any performance model (no matching template UID or no registered model)"
        )

    # Step 2: Apply filtering and reordering based on performance predictions
    return filter_and_sort_choices(ktc_stack, topk, discard_unranked)
