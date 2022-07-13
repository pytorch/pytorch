from typing import Dict

class CheckpointException(BaseException):
    """
    Exception raised if failure was detected as part of a checkpoint load or save.
    """
    def __init__(self, msg: str, failures: Dict[int, BaseException]):
        super().__init__(msg, failures)
        self._failures = failures

    @property
    def failures(self) -> Dict[int, BaseException]:
        """
        Returns:
            Dict of failed nodes and their associated exception.
              Keys are node ranks and values are exceptions
        """
        return self._failures
