set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

function install_huggingface() {
  local version
  version=$(get_pinned_commit huggingface)
  pip_install pandas
  pip_install scipy
  pip_install "transformers==${version}"
}

function install_timm() {
  local commit
  commit=$(get_pinned_commit timm)
  pip_install pandas
  pip_install scipy
  pip_install "git+https://github.com/rwightman/pytorch-image-models@${commit}"
}

function checkout_install_torchbench() {
  local commit
  commit=$(get_pinned_commit torchbench)
  git clone https://github.com/pytorch/benchmark torchbench
  pushd torchbench
  git checkout "$commit"

  if [ "$1" ]; then
    as_jenkins python install.py --continue_on_fail models "$@"
  else
    # Occasionally the installation may fail on one model but it is ok to continue
    # to install and test other models
    as_jenkins python install.py --continue_on_fail
  fi
  popd
}

function install_torchaudio() {
  local commit
  commit=$(get_pinned_commit audio)
  if [[ "$1" == "cuda" ]]; then
    # TODO: This is better to be passed as a parameter from _linux-test workflow
    # so that it can be consistent with what is set in build
    TORCH_CUDA_ARCH_LIST="8.0;8.6" pip_install --no-use-pep517 --user "git+https://github.com/pytorch/audio.git@${commit}"
  else
    pip_install --no-use-pep517 --user "git+https://github.com/pytorch/audio.git@${commit}"
  fi

}

function install_torchtext() {
  local data_commit
  local text_commit
  data_commit=$(get_pinned_commit data)
  text_commit=$(get_pinned_commit text)
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/data.git@${data_commit}"
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/text.git@${text_commit}"
}

function install_torchvision() {
  local commit
  commit=$(get_pinned_commit vision)
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/vision.git@${commit}"
}

install_huggingface
install_timm
install_torchaudio
install_torchtext
install_torchvision
# checkout_install_torchbench
