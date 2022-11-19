# =================== The following code **should** be executed inside Docker container ===================

# Install dependencies
sudo apt-get -y update
sudo apt-get -y install expect-dev

# This is where the local pytorch install in the docker image is located
pt_checkout="/var/lib/jenkins/workspace"

source "$pt_checkout/.jenkins/pytorch/common_utils.sh"

echo "dynamo_doc_push_script.sh: Invoked with $*"

set -ex

# for statements like ${1:-${DOCS_INSTALL_PATH:-docs/}}
# the order of operations goes:
#   1. Check if there's an argument $1
#   2. If no argument check for environment var DOCS_INSTALL_PATH
#   3. If no environment var fall back to default 'docs/'

# NOTE: It might seem weird to gather the second argument before gathering the first argument
#       but since DOCS_INSTALL_PATH can be derived from DOCS_VERSION it's probably better to
#       try and gather it first, just so we don't potentially break people who rely on this script
# Argument 2: What version of the docs we are building.
version="${2:-${DOCS_VERSION:-master}}"
if [ -z "$version" ]; then
echo "error: dynamo_doc_push_script.sh: version (arg2) not specified"
  exit 1
fi

# Argument 1: Where to copy the built documentation to
# (pytorch.github.io/$install_path)
install_path="${1:-${DOCS_INSTALL_PATH:-dynamo/${DOCS_VERSION}}}"
if [ -z "$install_path" ]; then
echo "error: dynamo_doc_push_script.sh: install_path (arg1) not specified"
  exit 1
fi

is_main_doc=false
if [ "$version" == "master" ]; then
  is_main_doc=true
fi

# Argument 3: The branch to push to. Usually is "site"
branch="${3:-${DOCS_BRANCH:-site}}"
if [ -z "$branch" ]; then
echo "error: dynamo_doc_push_script.sh: branch (arg3) not specified"
  exit 1
fi

echo "install_path: $install_path  version: $version"

# ======================== Building PyTorch C++ API Docs ========================



pushd "$pt_checkout/docs/torchdynamo"

pip -q install -r requirements.txt

make VERBOSE=1

popd

git clone https://github.com/pytorch/pytorch.github.io -b $branch --depth 1
pushd pytorch.github.io

git rm -rf "$install_path" || true
mv "$pt_checkout/docs/dynamo/build/html" "$install_path"

# Prevent Google from indexing $install_path/_modules. This folder contains
# generated source files.
# NB: the following only works on gnu sed. The sed shipped with mac os is different.
# One can `brew install gnu-sed` on a mac and then use "gsed" instead of "sed".
find "$install_path/_modules" -name "*.html" -print0 | xargs -0 sed -i '/<head>/a \ \ <meta name="robots" content="noindex">'

git add "$install_path" || true
git status
git config user.email "soumith+bot@pytorch.org"
git config user.name "pytorchbot"
# If there aren't changes, don't make a commit; push is no-op
git commit -m "Generate Python docs from pytorch/pytorch@${GITHUB_SHA}" || true
git status

git push -u origin HEAD:csl/test -f

popd
# =================== The above code **should** be executed inside Docker container ===================
