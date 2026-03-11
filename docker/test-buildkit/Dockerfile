# Minimal test image for validating BuildKit integration.
# Uses public.ecr.aws to avoid registry mirror dependencies.
FROM public.ecr.aws/docker/library/alpine:3.19

RUN echo "BuildKit integration test" > /README \
    && date -u > /BUILD_TIME
