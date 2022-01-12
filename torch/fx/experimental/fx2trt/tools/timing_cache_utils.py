import logging
import os

logger = logging.getLogger(__name__)


class TimingCacheManager:
    def __init__(self, timing_cache_prefix: str = "", save_timing_cache=False):
        # Setting timing cache for TRTInterpreter
        tc = os.environ.get("TRT_TIMING_CACHE_PREFIX", "")
        timing_cache_prefix_name = timing_cache_prefix
        if not timing_cache_prefix and tc:
            timing_cache_prefix_name = tc

        self.timing_cache_prefix_name = timing_cache_prefix
        self.save_timing_cache = save_timing_cache

    def get_file_full_name(self, name: str):
        return f"{self.timing_cache_prefix_name}_{name}.npy"

    def get_timing_cache_trt(self, timing_cache_file: str) -> bytearray:
        timing_cache_file = self.get_file_full_name(timing_cache_file)
        with open(timing_cache_file, "rb") as raw_cache:
            cache_data = raw_cache.read()
        return bytearray(cache_data)

    def update_timing_cache(
        self, timing_cache_file: str, serilized_cache: bytearray
    ) -> None:
        if not self.save_timing_cache:
            return
        timing_cache_file = self.get_file_full_name(timing_cache_file)
        with open(timing_cache_file, "wb") as local_cache:
            local_cache.seek(0)
            local_cache.write(serilized_cache)
            local_cache.truncate()
