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
    """Return True if a local Docker image exists."""
    if not image_name:
        return False

    client = client or _get_client()
    try:
        client.images.get(image_name)
        return True
    except (NotFound, APIError) as e:
        logger.error(
            "Error when checking Docker image '%s': %s",
            image_name,
            e.explanation if hasattr(e, "explanation") else str(e),
        )
        return False
