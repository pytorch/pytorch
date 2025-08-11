"""
Docker Utility helpers for CLI tasks.
"""

import logging
from typing import Optional

import docker
from docker.errors import APIError, NotFound


logger = logging.getLogger(__name__)

# lazy singleton so we don't reconnect every call
_docker_client: Optional[docker.DockerClient] = None


def _get_client() -> docker.DockerClient:
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def local_image_exists(
    image_name: str, client: Optional[docker.DockerClient] = None
) -> bool:
    """
    Return True if a local Docker image (by name:tag, id, or digest) exists.
    """
    if not image_name:
        return False
    client = client or _get_client()
    logger.info("Checking if image %s exists...", image_name)
    try:
        client.images.get(image_name)
        logger.info("Found %s...", image_name)
        return True
    except NotFound:
        logger.info("Image %s not found locally...", image_name)
        return False
    except APIError:
        logger.warning(
            "Run into apierror when trying to check docker image using docker client."
        )
        return False
