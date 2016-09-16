SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd $SCRIPT_DIR

python doc2md.py torch.nn --no-toc --all >../nn.md

popd
