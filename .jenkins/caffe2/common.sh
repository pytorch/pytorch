set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../.. && pwd)

# Figure out which Python to use
PYTHON="python"
if [[ "${BUILD_ENVIRONMENT}" =~ py((2|3)\.?[0-9]?\.?[0-9]?) ]]; then
  PYTHON="python${BASH_REMATCH[1]}"
fi

# Find where Caffe2 is installed. This will be the absolute path to the
# site-packages of the active Python installation
INSTALL_PREFIX="/usr/local/caffe2"
SITE_DIR=$($PYTHON -c "from distutils import sysconfig; print(sysconfig.get_python_lib(prefix=''))")
INSTALL_SITE_DIR="${INSTALL_PREFIX}/${SITE_DIR}"
CAFFE2_PYPATH="$INSTALL_SITE_DIR/caffe2"

# Set PYTHONPATH and LD_LIBRARY_PATH so that python can find the installed
# Caffe2.
export PYTHONPATH="${PYTHONPATH}:$INSTALL_SITE_DIR"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${INSTALL_PREFIX}/lib"
