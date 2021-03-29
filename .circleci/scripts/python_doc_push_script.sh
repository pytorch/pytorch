# =================== The following code **should** be executed inside Docker container ===================

# Install dependencies
sudo apt-get -y update
sudo apt-get -y install expect-dev

# This is where the local pytorch install in the docker image is located
pt_checkout="/var/lib/jenkins/workspace"

source "$pt_checkout/.jenkins/pytorch/common_utils.sh"

echo "python_doc_push_script.sh: Invoked with $*"

set -ex

# Argument 1: Where to copy the built documentation to
# (pytorch.github.io/$install_path)
install_path="$1"
if [ -z "$install_path" ]; then
echo "error: python_doc_push_script.sh: install_path (arg1) not specified"
  exit 1
fi

# Argument 2: What version of the docs we are building.
version="$2"
if [ -z "$version" ]; then
echo "error: python_doc_push_script.sh: version (arg2) not specified"
  exit 1
fi

is_master_doc=false
if [ "$version" == "master" ]; then
  is_master_doc=true
fi

# Argument 3: The branch to push to. Usually is "site"
branch="$3"
if [ -z "$branch" ]; then
echo "error: python_doc_push_script.sh: branch (arg3) not specified"
  exit 1
fi

echo "install_path: $install_path  version: $version"


build_docs () {
  set +e
  set -o pipefail
  make $1 2>&1 | tee /tmp/docs_build.txt
  code=$?
  if [ $code -ne 0 ]; then
    set +x
    echo =========================
    grep "WARNING:" /tmp/docs_build.txt
    echo =========================
    echo Docs build failed. If the failure is not clear, scan back in the log
    echo for any WARNINGS or for the line "build finished with problems"
    echo "(tried to echo the WARNINGS above the ==== line)"
    echo =========================
  fi
  set -ex
  return $code
}


git clone https://github.com/pytorch/pytorch.github.io -b $branch --depth 1
pushd pytorch.github.io

export LC_ALL=C
export PATH=/opt/conda/bin:$PATH

rm -rf pytorch || true

# Get all the documentation sources, put them in one place
pushd "$pt_checkout"
pushd docs

# Build the docs
pip -q install -r requirements.txt
if [ "$is_master_doc" = true ]; then
  build_docs html
  [ $? -eq 0 ] || exit $?
  make coverage
  # Now we have the coverage report, we need to make sure it is empty.
  # Count the number of lines in the file and turn that number into a variable
  # $lines. The `cut -f1 ...` is to only parse the number, not the filename
  # Skip the report header by subtracting 2: the header will be output even if
  # there are no undocumented items.
  #
  # Also: see docs/source/conf.py for "coverage_ignore*" items, which should
  # be documented then removed from there.
  lines=$(wc -l build/coverage/python.txt 2>/dev/null |cut -f1 -d' ')
  undocumented=$(($lines - 2))
  if [ $undocumented -lt 0 ]; then
    echo coverage output not found
    exit 1
  elif [ $undocumented -gt 0 ]; then
    echo undocumented objects found:
    cat build/coverage/python.txt
    exit 1
  fi
else
  # skip coverage, format for stable or tags
  build_docs html-stable
  [ $? -eq 0 ] || exit $?
fi

# Move them into the docs repo
popd
popd
git rm -rf "$install_path" || true
mv "$pt_checkout/docs/build/html" "$install_path"

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
git commit -m "Generate Python docs from pytorch/pytorch@$CIRCLE_SHA1" || true
git status

popd
# =================== The above code **should** be executed inside Docker container ===================
