import os

from pathlib import Path

# Workaround for compile flags ending up in the link interface for XNNPACK.
# TODO Remove this after landing fix upstream and updating.
TO_PATCH = "TARGET_LINK_LIBRARIES(XNNPACK PUBLIC xnnpack-base)"
PATCH_REPLACEMENT = "TARGET_LINK_LIBRARIES(XNNPACK PRIVATE xnnpack-base)"

def patch_xnnpack_cmake():
    xnnpack_cmakelists_path = Path(os.path.dirname(os.path.realpath(__file__))).parent / "third_party" / "XNNPACK" / "CMakeLists.txt"
    with open(xnnpack_cmakelists_path, "r") as f:
        contents = f.read()
        contents = contents.replace(TO_PATCH, PATCH_REPLACEMENT)
        
    with open(xnnpack_cmakelists_path, "w") as f:
        f.write(contents)

if __name__ == "__main__":
    patch_xnnpack_cmake()
