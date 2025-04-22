import os


# Whether to disable showing progress on compilation passes
# Need to add a new config otherwise wil get a circular import if dynamo config is imported here
disable_progress = True

# If True this also shows the node names in each pass, for small models this is great but larger models it's quite noisy
verbose_progress = False

# Feature flag to control whether source location information (file_path, line_number) is captured.
capture_source_locations: bool = (
    os.getenv("TORCH_FX_CAPTURE_SOURCE_LOCATIONS", "0") == "1"
)
