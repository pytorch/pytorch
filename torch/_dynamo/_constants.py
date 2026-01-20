"""
Shared constants for TorchDynamo internals.
This module should have no imports from other _dynamo modules to avoid circular dependencies.
"""

# Field names for tracking dynamic dimension information on tensors
SHAPE_IDS_FIELD_NAME = "_dynamo_shape_ids"
UNBACKED_INDICES_FIELD_NAME = "_dynamo_unbacked_indices"
DYNAMIC_INDICES_FIELD_NAME = "_dynamo_dynamic_indices"
