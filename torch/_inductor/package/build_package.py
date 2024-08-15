build_package_contents = """
import os
from pathlib import Path

from torch._inductor.package.package import compile_so

curr_dir = Path(__file__).parent
aoti_files = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(curr_dir)
    for file in files
]

output_so = compile_so(curr_dir, aoti_files, curr_dir)
"""
