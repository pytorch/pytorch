"""NumPy helper.

Note: If you plan to add a library detection script like this one, consider it twice. Most library detection should go
to CMake script. This one is an exception, because Python code can do a much better job due to NumPy's inherent Pythonic
nature.
"""

from .env import check_negative_env_flag


# Set USE_NUMPY to what the user wants, because even if we fail here, cmake
# will check for the presence of NumPy again (`cmake/Dependencies.cmake`).
USE_NUMPY = not check_negative_env_flag("USE_NUMPY")
