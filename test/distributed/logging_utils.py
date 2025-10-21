import logging
import time


_start_time = time.time()
_logger = logging.getLogger(__name__)


def _ts():
    return time.time() - _start_time


def configure(level=logging.INFO, force=False):
    try:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(name)s %(levelname)s: %(message)s",
            force=force,
        )
    except TypeError:
        logging.basicConfig(
            level=level, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
        )


def log_test_info(rank, message):
    _logger.info("[%7.3fs][Rank %s] %s", _ts(), rank, message)


def log_test_success(rank, message):
    _logger.info("[%7.3fs][Rank %s] ✅ %s", _ts(), rank, message)


def log_test_validation(rank, message):
    _logger.info("[%7.3fs][Rank %s] ✓ %s", _ts(), rank, message)


def log_test_warning(rank, message):
    _logger.warning("[%7.3fs][Rank %s] ⚠️ %s", _ts(), rank, message)


def log_test_error(rank, message):
    _logger.error("[%7.3fs][Rank %s] ✗ %s", _ts(), rank, message)
