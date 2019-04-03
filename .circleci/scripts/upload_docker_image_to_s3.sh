DOCKER_IMAGE_BASE_NAME=$(basename "${DOCKER_IMAGE}")
export AWS_ACCESS_KEY_ID=${CIRCLECI_AWS_ACCESS_KEY_FOR_OSSCI_LINUX_BUILD_S3_BUCKET_V1}
export AWS_SECRET_ACCESS_KEY=${CIRCLECI_AWS_SECRET_KEY_FOR_OSSCI_LINUX_BUILD_S3_BUCKET_V1}
set +e
retcode=$(aws s3 ls s3://ossci-linux-build/${DOCKER_IMAGE_TYPE}/${DOCKER_IMAGE_BASE_NAME}.tar)
set -e
if [ -z "$retcode" ]; then
  docker save -o "${DOCKER_IMAGE_BASE_NAME}.tar" "${DOCKER_IMAGE}"
  aws s3 cp "${DOCKER_IMAGE_BASE_NAME}.tar" "s3://ossci-linux-build/${DOCKER_IMAGE_TYPE}/${DOCKER_IMAGE_BASE_NAME}.tar" --acl public-read --only-show-errors
  echo "Docker image is uploaded to S3."
  rm "${DOCKER_IMAGE_BASE_NAME}.tar"
fi
echo "Docker image download link: https://s3.amazonaws.com/ossci-linux-build/${DOCKER_IMAGE_TYPE}/${DOCKER_IMAGE_BASE_NAME}.tar. To use this image, please follow the instructions at https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md#reproducing-linux-ci-errors-locally."
