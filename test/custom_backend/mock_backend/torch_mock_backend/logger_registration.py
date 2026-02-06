from torch._logging._internal import getArtifactLogger, register_artifact, register_log


register_log("test_component", ["torch_mock_backend.logger_registration"])
register_artifact(
    "test_artifact",
    "torch mock backend artifact.",
)


def test_logger():
    import logging

    commponent_logger = logging.getLogger(__name__)
    commponent_logger.info("custom backend component info log")

    artifact_logger = getArtifactLogger(__name__, "test_artifact")
    artifact_logger.info("custom backend artifact info log")
