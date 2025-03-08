# Owner(s): ["module: unknown"]
import concurrent.futures
import tempfile
import time

from torch.testing._internal.common_utils import run_tests, skipIfWindows, TestCase
from torch.utils._filelock import FileLock


class TestFileLock(TestCase):
    def test_no_crash(self):
        _, p = tempfile.mkstemp()
        with FileLock(p):
            pass

    @skipIfWindows(
        msg="Windows doesn't support multiple files being opened at once easily"
    )
    def test_sequencing(self):
        with tempfile.NamedTemporaryFile() as ofd:
            p = ofd.name

            def test_thread(i):
                with FileLock(p + ".lock"):
                    start = time.time()
                    with open(p, "a") as fd:
                        fd.write(str(i))
                    end = time.time()
                    return (start, end)

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(test_thread, i) for i in range(10)]
                times = []
                for f in futures:
                    times.append(f.result(60))

            with open(p) as fd:
                self.assertEqual(
                    set(fd.read()), {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
                )

            for i, (start, end) in enumerate(times):
                for j, (newstart, newend) in enumerate(times):
                    if i == j:
                        continue

                    # Times should never intersect
                    self.assertFalse(newstart > start and newstart < end)
                    self.assertFalse(newend > start and newstart < end)


if __name__ == "__main__":
    run_tests()
