import logging


def get_logger(app=None, name="torch_cli.lib"):
    """
     used in lib to either use app logger or fallback to standard python logger
     example:
        from lib.logger import get_logger

        def do_work(app=None):
            log = get_logger(app)
            log.info("Doing work in utils...")
    """
    if app and hasattr(app, "log"):
        return app.log

    # Fallback to standard Python logger
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
