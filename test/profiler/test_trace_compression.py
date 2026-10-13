# Owner(s): ["oncall: profiler"]

import gzip
import io
import json

from torch.profiler._trace_compression import _get_trace_file_opener
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestTraceCompression(TestCase):
    @parametrize("suffix", [".gz", ".zst"])
    @parametrize("compresslevel", [None, 0, 1, 6, 9])
    def test_compression_level(self, suffix, compresslevel):
        events = [{"name": f"event {i % 10}", "ts": i} for i in range(1000)]
        data = json.dumps({"traceEvents": events}).encode()
        expected_buffer = io.BytesIO()
        if suffix == ".gz":
            level = 9 if compresslevel is None else compresslevel
            writer = gzip.GzipFile(
                fileobj=expected_buffer, mode="wb", compresslevel=level, mtime=0
            )
        else:
            try:
                import zstandard
            except ImportError:
                self.skipTest("requires zstandard")
            level = 1 if compresslevel is None else compresslevel
            compressor = zstandard.ZstdCompressor(level=level)
            writer = compressor.stream_writer(expected_buffer, closefd=False)

        with writer as stream:
            stream.write(data)
        expected = expected_buffer.getvalue()

        opener = _get_trace_file_opener("trace.json" + suffix, compresslevel=compresslevel)
        buffer = io.BytesIO()
        with opener(buffer, "wb") as stream:
            stream.write(data)
        compressed = buffer.getvalue()
        if suffix == ".gz":
            # Ignore the gzip header, which includes a timestamp.
            self.assertEqual(compressed[10:], expected[10:])
            self.assertEqual(gzip.decompress(compressed), data)
        else:
            self.assertEqual(compressed, expected)
            with zstandard.open(io.BytesIO(compressed), "rb") as stream:
                self.assertEqual(stream.read(), data)


if __name__ == "__main__":
    run_tests()
