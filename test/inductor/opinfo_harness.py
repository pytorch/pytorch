import os
import subprocess

from torch.testing._internal.common_methods_invocations import op_db

if __name__ == "__main__":
    i = 0
    while i < len(op_db):
        start = i
        end = i + 20
        os.environ["PYTORCH_TEST_RANGE_START"] = f"{start}"
        os.environ["PYTORCH_TEST_RANGE_END"] = f"{end}"
        popen = subprocess.Popen(
            ["pytest", "test/inductor/test_torchinductor_opinfo.py"],
            stdout=subprocess.PIPE,
        )
        for line in popen.stdout:
            print(line.decode(), end="")
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(
                return_code, ["pytest", "test/inductor/test_torchinductor_opinfo.py"]
            )
        i = end + 1
