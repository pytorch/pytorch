export MAX_JOBS=16
pip uninstall torch -y

# BUILD_ENVIRONMENT is set in rocm docker conatiners
bash .jenkins/pytorch/build.sh