import re
from typing import Dict, List

import boto3  # type: ignore[import-untyped]


S3 = boto3.resource("s3")
CLIENT = boto3.client("s3")
BUCKET = S3.Bucket("pytorch")

PACKAGES_PER_PROJECT: Dict[str, List[Dict[str, str]]] = {
    "sympy": [{"version": "latest", "project": "torch"}],
    "mpmath": [{"version": "latest", "project": "torch"}],
    "pillow": [{"version": "latest", "project": "torch"}],
    "networkx": [{"version": "latest", "project": "torch"}],
    "numpy": [{"version": "latest", "project": "torch"}],
    "jinja2": [{"version": "latest", "project": "torch"}],
    "filelock": [{"version": "latest", "project": "torch"}],
    "fsspec": [{"version": "latest", "project": "torch"}],
    "nvidia-cudnn-cu11": [{"version": "latest", "project": "torch"}],
    "typing-extensions": [{"version": "latest", "project": "torch"}],
    "nvidia-cuda-nvrtc-cu12": [
        {
            "version": "12.6.77",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "12.8.93",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "12.9.86",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-cuda-nvrtc": [
        {
            "version": "13.0.88",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-cuda-runtime-cu12": [
        {
            "version": "12.6.77",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "12.8.90",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "12.9.79",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-cuda-runtime": [
        {
            "version": "13.0.96",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-cuda-cupti-cu12": [
        {
            "version": "12.6.80",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "12.8.90",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "12.9.79",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-cuda-cupti": [
        {
            "version": "13.0.85",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-cudnn-cu12": [
        {
            "version": "9.10.2.21",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "9.10.2.21",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "9.10.2.21",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-cudnn-cu13": [
        {
            "version": "9.13.0.50",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-cublas-cu12": [
        {
            "version": "12.6.4.1",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "12.8.4.1",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "12.9.1.4",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-cublas": [
        {
            "version": "13.1.0.3",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-cufft-cu12": [
        {
            "version": "11.3.0.4",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "11.3.3.83",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "11.4.1.4",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-cufft": [
        {
            "version": "12.0.0.61",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-curand-cu12": [
        {
            "version": "10.3.7.77",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "10.3.9.90",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "10.3.10.19",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-curand": [
        {
            "version": "10.4.0.42",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-cusolver-cu12": [
        {
            "version": "11.7.1.2",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "11.7.3.90",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "11.7.5.82",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-cusolver": [
        {
            "version": "12.0.4.66",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-cusparse-cu12": [
        {
            "version": "12.5.4.2",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "12.5.8.93",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "12.5.10.65",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-cusparse": [
        {
            "version": "12.6.3.3",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-cusparselt-cu12": [
        {
            "version": "0.7.1",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "0.7.1",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "0.7.1",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-cusparselt-cu13": [
        {
            "version": "0.8.0",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-nccl-cu12": [
        {
            "version": "2.27.5",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "2.27.5",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "2.27.5",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-nccl-cu13": [
        {
            "version": "2.27.7",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-nvshmem-cu12": [
        {
            "version": "3.4.5",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "3.4.5",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "3.4.5",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-nvshmem-cu13": [
        {
            "version": "3.4.5",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-nvtx-cu12": [
        {
            "version": "12.6.77",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "12.8.90",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "12.9.79",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-nvtx": [
        {
            "version": "13.0.85",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-nvjitlink-cu12": [
        {
            "version": "12.6.85",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "12.8.93",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "12.9.86",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-nvjitlink": [
        {
            "version": "13.0.88",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "nvidia-cufile-cu12": [
        {
            "version": "1.11.1.6",
            "project": "torch",
            "target": "cu126",
        },
        {
            "version": "1.13.1.3",
            "project": "torch",
            "target": "cu128",
        },
        {
            "version": "1.14.1.1",
            "project": "torch",
            "target": "cu129",
        },
    ],
    "nvidia-cufile": [
        {
            "version": "1.15.1.6",
            "project": "torch",
            "target": "cu130",
        }
    ],
    "arpeggio": [{"version": "latest", "project": "triton"}],
    "caliper-reader": [{"version": "latest", "project": "triton"}],
    "contourpy": [{"version": "latest", "project": "triton"}],
    "cycler": [{"version": "latest", "project": "triton"}],
    "dill": [{"version": "latest", "project": "triton"}],
    "fonttools": [{"version": "latest", "project": "triton"}],
    "kiwisolver": [{"version": "latest", "project": "triton"}],
    "llnl-hatchet": [{"version": "latest", "project": "triton"}],
    "matplotlib": [{"version": "latest", "project": "triton"}],
    "pandas": [{"version": "latest", "project": "triton"}],
    "pydot": [{"version": "latest", "project": "triton"}],
    "pyparsing": [{"version": "latest", "project": "triton"}],
    "pytz": [{"version": "latest", "project": "triton"}],
    "textX": [{"version": "latest", "project": "triton"}],
    "tzdata": [{"version": "latest", "project": "triton"}],
    "importlib-metadata": [{"version": "latest", "project": "triton"}],
    "importlib-resources": [{"version": "latest", "project": "triton"}],
    "zipp": [{"version": "latest", "project": "triton"}],
    "aiohttp": [{"version": "latest", "project": "torchtune"}],
    "aiosignal": [{"version": "latest", "project": "torchtune"}],
    "antlr4-python3-runtime": [{"version": "latest", "project": "torchtune"}],
    "attrs": [{"version": "latest", "project": "torchtune"}],
    "blobfile": [{"version": "latest", "project": "torchtune"}],
    "certifi": [{"version": "latest", "project": "torchtune"}],
    "charset-normalizer": [{"version": "latest", "project": "torchtune"}],
    "datasets": [{"version": "latest", "project": "torchtune"}],
    "frozenlist": [{"version": "latest", "project": "torchtune"}],
    "huggingface-hub": [{"version": "latest", "project": "torchtune"}],
    "idna": [{"version": "latest", "project": "torchtune"}],
    "lxml": [{"version": "latest", "project": "torchtune"}],
    "markupsafe": [{"version": "latest", "project": "torchtune"}],
    "multidict": [{"version": "latest", "project": "torchtune"}],
    "multiprocess": [{"version": "latest", "project": "torchtune"}],
    "omegaconf": [{"version": "latest", "project": "torchtune"}],
    "pyarrow": [{"version": "latest", "project": "torchtune"}],
    "pyarrow-hotfix": [{"version": "latest", "project": "torchtune"}],
    "pycryptodomex": [{"version": "latest", "project": "torchtune"}],
    "python-dateutil": [{"version": "latest", "project": "torchtune"}],
    "pyyaml": [{"version": "latest", "project": "torchtune"}],
    "regex": [{"version": "latest", "project": "torchtune"}],
    "requests": [{"version": "latest", "project": "torchtune"}],
    "safetensors": [{"version": "latest", "project": "torchtune"}],
    "sentencepiece": [{"version": "latest", "project": "torchtune"}],
    "six": [{"version": "latest", "project": "torchtune"}],
    "tiktoken": [{"version": "latest", "project": "torchtune"}],
    "tqdm": [{"version": "latest", "project": "torchtune"}],
    "urllib3": [{"version": "latest", "project": "torchtune"}],
    "xxhash": [{"version": "latest", "project": "torchtune"}],
    "yarl": [{"version": "latest", "project": "torchtune"}],
    "dpcpp-cpp-rt": [{"version": "latest", "project": "torch_xpu"}],
    "intel-cmplr-lib-rt": [{"version": "latest", "project": "torch_xpu"}],
    "intel-cmplr-lib-ur": [{"version": "latest", "project": "torch_xpu"}],
    "intel-cmplr-lic-rt": [{"version": "latest", "project": "torch_xpu"}],
    "intel-opencl-rt": [{"version": "latest", "project": "torch_xpu"}],
    "intel-sycl-rt": [{"version": "latest", "project": "torch_xpu"}],
    "intel-openmp": [{"version": "latest", "project": "torch_xpu"}],
    "tcmlib": [{"version": "latest", "project": "torch_xpu"}],
    "umf": [{"version": "latest", "project": "torch_xpu"}],
    "intel-pti": [{"version": "latest", "project": "torch_xpu"}],
    "tbb": [{"version": "latest", "project": "torch_xpu"}],
    "oneccl-devel": [{"version": "latest", "project": "torch_xpu"}],
    "oneccl": [{"version": "latest", "project": "torch_xpu"}],
    "impi-rt": [{"version": "latest", "project": "torch_xpu"}],
    "onemkl-sycl-blas": [{"version": "latest", "project": "torch_xpu"}],
    "onemkl-sycl-dft": [{"version": "latest", "project": "torch_xpu"}],
    "onemkl-sycl-lapack": [{"version": "latest", "project": "torch_xpu"}],
    "onemkl-sycl-sparse": [{"version": "latest", "project": "torch_xpu"}],
    "onemkl-sycl-rng": [{"version": "latest", "project": "torch_xpu"}],
    "mkl": [{"version": "latest", "project": "torch_xpu"}],
    # vLLM
    "ninja": [{"version": "latest", "project": "vllm"}],
    "cuda-python": [{"version": "12.9.0", "project": "vllm"}],
    "cuda-bindings": [{"version": "12.9.2", "project": "vllm"}],
    "cuda-pathfinder": [{"version": "latest", "project": "vllm"}],
    "pynvml": [{"version": "latest", "project": "vllm"}],
    "nvidia-ml-py": [{"version": "latest", "project": "vllm"}],
    "einops": [{"version": "latest", "project": "vllm"}],
    "packaging": [{"version": "latest", "project": "vllm"}],
    "nvidia-cudnn-frontend": [{"version": "latest", "project": "vllm"}],
    "cachetools": [{"version": "latest", "project": "vllm"}],
    "blake3": [{"version": "latest", "project": "vllm"}],
    "py-cpuinfo": [{"version": "latest", "project": "vllm"}],
    "transformers": [{"version": "latest", "project": "vllm"}],
    "hf-xet": [{"version": "latest", "project": "vllm"}],
    "tokenizers": [{"version": "latest", "project": "vllm"}],
    "protobuf": [{"version": "latest", "project": "vllm"}],
    "fastapi": [{"version": "latest", "project": "vllm"}],
    "annotated-types": [{"version": "latest", "project": "vllm"}],
    "anyio": [{"version": "latest", "project": "vllm"}],
    "pydantic": [{"version": "latest", "project": "vllm"}],
    "pydantic-core": [{"version": "2.33.2", "project": "vllm"}],
    "sniffio": [{"version": "latest", "project": "vllm"}],
    "starlette": [{"version": "latest", "project": "vllm"}],
    "typing-inspection": [{"version": "latest", "project": "vllm"}],
    "openai": [{"version": "latest", "project": "vllm"}],
    "distro": [{"version": "latest", "project": "vllm"}],
    "h11": [{"version": "latest", "project": "vllm"}],
    "httpcore": [{"version": "latest", "project": "vllm"}],
    "httpx": [{"version": "latest", "project": "vllm"}],
    "jiter": [{"version": "latest", "project": "vllm"}],
    "prometheus-client": [{"version": "latest", "project": "vllm"}],
    "prometheus-fastapi-instrumentator": [{"version": "latest", "project": "vllm"}],
    "lm-format-enforcer": [{"version": "latest", "project": "vllm"}],
    "interegular": [{"version": "latest", "project": "vllm"}],
    "llguidance": [{"version": "0.7.11", "project": "vllm"}],
    "outlines-core": [{"version": "0.2.10", "project": "vllm"}],
    "diskcache": [{"version": "latest", "project": "vllm"}],
    "lark": [{"version": "latest", "project": "vllm"}],
    "xgrammar": [{"version": "0.1.23", "project": "vllm"}],
    "partial-json-parser": [{"version": "latest", "project": "vllm"}],
    "pyzmq": [{"version": "latest", "project": "vllm"}],
    "msgspec": [{"version": "latest", "project": "vllm"}],
    "gguf": [{"version": "latest", "project": "vllm"}],
    "mistral-common": [{"version": "latest", "project": "vllm"}],
    "rpds-py": [{"version": "latest", "project": "vllm"}],
    "pycountry": [{"version": "latest", "project": "vllm"}],
    "referencing": [{"version": "latest", "project": "vllm"}],
    "pydantic-extra-types": [{"version": "latest", "project": "vllm"}],
    "jsonschema-specifications": [{"version": "latest", "project": "vllm"}],
    "jsonschema": [{"version": "latest", "project": "vllm"}],
    "opencv-python-headless": [{"version": "latest", "project": "vllm"}],
    "compressed-tensors": [{"version": "latest", "project": "vllm"}],
    "frozendict": [{"version": "latest", "project": "vllm"}],
    "depyf": [{"version": "latest", "project": "vllm"}],
    "astor": [{"version": "latest", "project": "vllm"}],
    "cloudpickle": [{"version": "latest", "project": "vllm"}],
    "watchfiles": [{"version": "latest", "project": "vllm"}],
    "python-json-logger": [{"version": "latest", "project": "vllm"}],
    "scipy": [{"version": "latest", "project": "vllm"}],
    "pybase64": [{"version": "latest", "project": "vllm"}],
    "cbor2": [{"version": "latest", "project": "vllm"}],
    "setproctitle": [{"version": "latest", "project": "vllm"}],
    "openai-harmony": [{"version": "latest", "project": "vllm"}],
    "numba": [{"version": "0.61.2", "project": "vllm"}],
    "llvmlite": [{"version": "latest", "project": "vllm"}],
    "ray": [{"version": "latest", "project": "vllm"}],
    "click": [{"version": "latest", "project": "vllm"}],
    "msgpack": [{"version": "latest", "project": "vllm"}],
    "fastapi-cli": [{"version": "latest", "project": "vllm"}],
    "httptools": [{"version": "latest", "project": "vllm"}],
    "markdown-it-py": [{"version": "latest", "project": "vllm"}],
    "pygments": [{"version": "latest", "project": "vllm"}],
    "python-dotenv": [{"version": "latest", "project": "vllm"}],
    "rich": [{"version": "latest", "project": "vllm"}],
    "rich-toolkit": [{"version": "latest", "project": "vllm"}],
    "shellingham": [{"version": "latest", "project": "vllm"}],
    "typer": [{"version": "latest", "project": "vllm"}],
    "uvicorn": [{"version": "latest", "project": "vllm"}],
    "uvloop": [{"version": "latest", "project": "vllm"}],
    "websockets": [{"version": "latest", "project": "vllm"}],
    "python-multipart": [{"version": "latest", "project": "vllm"}],
    "email-validator": [{"version": "latest", "project": "vllm"}],
    "dnspython": [{"version": "2.7.0", "project": "vllm"}],
    "fastapi-cloud-cli": [{"version": "latest", "project": "vllm"}],
    "mdurl": [{"version": "latest", "project": "vllm"}],
    "rignore": [{"version": "latest", "project": "vllm"}],
    "sentry-sdk": [{"version": "latest", "project": "vllm"}],
    "cupy-cuda12x": [{"version": "latest", "project": "vllm"}],
    "fastrlock": [{"version": "latest", "project": "vllm"}],
    "soundfile": [{"version": "latest", "project": "vllm"}],
    "cffi": [{"version": "latest", "project": "vllm"}],
    "pycparser": [{"version": "latest", "project": "vllm"}],
}


def download(url: str) -> bytes:
    from urllib.request import urlopen

    with urlopen(url) as conn:
        return conn.read()


def is_stable(package_version: str) -> bool:
    return bool(re.match(r"^([0-9]+\.)+[0-9]+$", package_version))


def is_nvidia_package(pkg_name: str) -> bool:
    """Check if a package is an NVIDIA CUDA package."""
    return pkg_name.startswith("nvidia-") or pkg_name.startswith("cuda-")


def parse_simple_idx(url: str) -> Dict[str, str]:
    html = download(url).decode("ascii")
    return {
        name: url
        for (url, name) in re.findall('<a href="([^"]+)"[^>]*>([^>]+)</a>', html)
    }


def get_whl_versions(idx: Dict[str, str]) -> List[str]:
    return [
        k.split("-")[1]
        for k in idx.keys()
        if k.endswith(".whl") and is_stable(k.split("-")[1])
    ]


def get_wheels_of_version(idx: Dict[str, str], version: str) -> Dict[str, str]:
    return {
        k: v
        for (k, v) in idx.items()
        if k.endswith(".whl") and k.split("-")[1] == version
    }


def convert_relative_to_absolute_links(html: str, base_url: str) -> str:
    """
    Convert relative links in HTML to absolute links.

    Args:
        html: HTML content with relative links
        base_url: Base URL to prepend to relative links (e.g., https://pypi.nvidia.com/nvidia-cudnn-cu11/)

    Returns:
        Modified HTML with absolute links
    """
    # Pattern to match href attributes with relative links
    # This will match: href="filename.whl#sha256=..."
    def replace_href(match):
        href_content = match.group(1)
        # Only modify if it's a relative link (doesn't start with http:// or https://)
        if not href_content.startswith(('http://', 'https://')):
            # Prepend base URL to make it absolute
            return f'href="{base_url}{href_content}"'
        return match.group(0)

    # Replace relative links with absolute links
    modified_html = re.sub(r'href="([^"]+)"', replace_href, html)
    return modified_html


def upload_nvidia_index(
    pkg_name: str,
    prefix: str = "whl/test",
    *,
    dry_run: bool = False,
) -> None:
    """
    Download HTML index from pypi.nvidia.com, convert relative links to absolute,
    and upload to S3.

    Args:
        pkg_name: Name of the NVIDIA package
        prefix: S3 prefix path (e.g., whl/test or whl/test/cu126)
        dry_run: If True, don't actually upload to S3
    """
    nvidia_pypi_url = f"https://pypi.nvidia.com/{pkg_name}/"

    try:
        print(f"Downloading index for {pkg_name} from {nvidia_pypi_url}")
        html = download(nvidia_pypi_url).decode("utf-8")

        # Convert relative links to absolute links
        modified_html = convert_relative_to_absolute_links(html, nvidia_pypi_url)

        # Upload to S3
        s3_key = f"{prefix}/{pkg_name}/index.html"

        if dry_run:
            print(f"Dry Run - Would upload index.html to s3://pytorch/{s3_key}")
            print(f"Preview of first 500 chars:\n{modified_html[:500]}")
        else:
            print(f"Uploading index.html to s3://pytorch/{s3_key}")
            BUCKET.Object(key=s3_key).put(
                ACL="public-read",
                ContentType="text/html",
                Body=modified_html.encode("utf-8")
            )
            print(f"Successfully uploaded index for {pkg_name}")

    except Exception as e:
        print(f"Error processing {pkg_name}: {e}")


def upload_missing_whls(
    pkg_name: str = "numpy",
    prefix: str = "whl/test",
    *,
    dry_run: bool = False,
    only_pypi: bool = False,
    target_version: str = "latest",
) -> None:
    # For NVIDIA packages, use the new indexing method
    if is_nvidia_package(pkg_name):
        upload_nvidia_index(pkg_name, prefix, dry_run=dry_run)
        return

    # For non-NVIDIA packages, use the original PyPI logic
    pypi_idx = parse_simple_idx(f"https://pypi.org/simple/{pkg_name}")
    pypi_versions = get_whl_versions(pypi_idx)

    # Determine which version to use
    if target_version == "latest" or not target_version:
        selected_version = pypi_versions[-1] if pypi_versions else None
    elif target_version in pypi_versions:
        selected_version = target_version
    else:
        print(
            f"Warning: Version {target_version} not found for {pkg_name}, using latest"
        )
        selected_version = pypi_versions[-1] if pypi_versions else None

    if not selected_version:
        print(f"No stable versions found for {pkg_name}")
        return

    pypi_latest_packages = get_wheels_of_version(pypi_idx, selected_version)

    download_latest_packages: Dict[str, str] = {}
    if not only_pypi:
        download_idx = parse_simple_idx(
            f"https://download.pytorch.org/{prefix}/{pkg_name}"
        )
        download_latest_packages = get_wheels_of_version(download_idx, selected_version)

    has_updates = False
    for pkg in pypi_latest_packages:
        if pkg in download_latest_packages:
            continue
        # Skip pp packages
        if "-pp3" in pkg:
            continue
        # Skip win32 packages
        if "-win32" in pkg:
            continue
        # Skip muslinux packages
        if "-musllinux" in pkg:
            continue
        print(f"Downloading {pkg}")
        if dry_run:
            has_updates = True
            print(f"Dry Run - not Uploading {pkg} to s3://pytorch/{prefix}/")
            continue
        data = download(pypi_idx[pkg])
        print(f"Uploading {pkg} to s3://pytorch/{prefix}/")
        BUCKET.Object(key=f"{prefix}/{pkg}").put(
            ACL="public-read", ContentType="binary/octet-stream", Body=data
        )
        has_updates = True
    if not has_updates:
        print(f"{pkg_name} is already at version {selected_version} for {prefix}")


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser("Upload dependent packages to s3://pytorch")
    # Get unique paths from the packages list
    project_paths = list(
        {
            config["project"]
            for pkg_configs in PACKAGES_PER_PROJECT.values()
            for config in pkg_configs
        }
    )
    project_paths += ["all"]
    parser.add_argument("--package", choices=project_paths, default="torch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-pypi", action="store_true")
    parser.add_argument("--include-stable", action="store_true")
    args = parser.parse_args()

    SUBFOLDERS = ["whl/nightly", "whl/test"]
    if args.include_stable:
        SUBFOLDERS.append("whl")

    for prefix in SUBFOLDERS:
        # Process each package and its multiple configurations
        for pkg_name, pkg_configs in PACKAGES_PER_PROJECT.items():
            # Filter configurations by the selected project
            selected_configs = [
                config
                for config in pkg_configs
                if args.package == "all" or config["project"] == args.package
            ]

            # Process each configuration for this package
            for pkg_config in selected_configs:
                if "target" in pkg_config and pkg_config["target"] != "":
                    full_path = f"{prefix}/{pkg_config['target']}"
                else:
                    full_path = f"{prefix}"

                upload_missing_whls(
                    pkg_name,
                    full_path,
                    dry_run=args.dry_run,
                    only_pypi=args.only_pypi,
                    target_version=pkg_config["version"],
                )


if __name__ == "__main__":
    main()
