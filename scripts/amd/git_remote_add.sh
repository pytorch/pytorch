GITHUB_URL=https://github.com/pytorch/pytorch
REPO_NAME=$(basename $GITHUB_URL)

git remote add upstream $GITHUB_URL
git remote add rocm_fork https://github.com/ROCmSoftwarePlatform/$REPO_NAME
git remote add private_fork https://github.com/micmelesse/$REPO_NAME
