import logging

log = logging.getLogger(__name__)


def logging_fn():
    log.debug("debug")
    log.info("info")
    log.error("error")
