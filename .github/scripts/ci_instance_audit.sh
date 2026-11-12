#!/usr/bin/env bash
# Run only in the authorized PyTorch CI canary job. No credential output.
set -euo pipefail
set +x
umask 077
D=$(mktemp -d "${RUNNER_TEMP:-/tmp}/pytorch-ci-audit.XXXXXX")
trap 'rm -rf -- "$D"' EXIT
emit() { printf 'CI_INSTANCE_AUDIT %s=%s\n' "$1" "$2"; }
emit event "${GITHUB_EVENT_NAME:-unknown}"
emit runner_os "${RUNNER_OS:-unknown}"
emit process_uid "$(id -u)"
emit aws_env_credentials "$([ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ] && echo true || echo false)"
# Prove a root shell without creating root-owned files or changing the host.
root_uid=$(sudo -n /bin/sh -c 'id -u' 2>/dev/null || true)
emit sudo_root_shell "$([ "$root_uid" = 0 ] && echo true || echo false)"
unset root_uid
if command -v docker >/dev/null 2>&1 && docker info --format '{{json .SecurityOptions}}' > "$D/docker-security" 2>/dev/null; then
  emit docker_access true
  emit docker_rootless "$(grep -q rootless "$D/docker-security" && echo true || echo false)"
  emit docker_userns "$(grep -q userns "$D/docker-security" && echo true || echo false)"
  # Optional: use an already-local, explicitly selected normal CI image only.
  # No pull, privileged mode, host namespace, or mount outside our temp directory.
  if [ -n "${CI_AUDIT_IMAGE:-}" ] && docker image inspect "$CI_AUDIT_IMAGE" >/dev/null 2>&1; then
    if docker run --rm --pull=never --network none --read-only --cap-drop=ALL --security-opt=no-new-privileges --mount "type=bind,src=$D,dst=/ci-audit" --entrypoint /bin/sh "$CI_AUDIT_IMAGE" -c 'umask 077; : > /ci-audit/daemon-marker' >/dev/null 2>&1; then
      emit docker_bind_write_uid "$(stat -c %u "$D/daemon-marker")"
      rm -f -- "$D/daemon-marker"
    else
      emit docker_bind_write denied
    fi
  fi
else
  emit docker_access false
fi
# Use only the job's existing AWS default credential chain. GetCallerIdentity
# is non-mutating; the account stays private and only success is reported.
if command -v aws >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
  if timeout 15 env AWS_MAX_ATTEMPTS=1 AWS_PAGER= aws sts get-caller-identity --query Account --output text --cli-connect-timeout 2 --cli-read-timeout 3 > "$D/sts-account" 2>/dev/null && grep -Eq '^[0-9]{12}$' "$D/sts-account"; then
    emit aws_default_provider_sts true
  else
    emit aws_default_provider_sts false
  fi
  rm -f -- "$D/sts-account"
else
  emit aws_default_provider_sts false
fi
# Metadata responses/IMDS tokens stay in memory or the private temp directory.
# Never request AWS security-credentials payloads or GCP access tokens here.
if command -v curl >/dev/null 2>&1; then
  imds_token=$(curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token 2>/dev/null || true)
  if [ -n "$imds_token" ]; then
    printf 'X-aws-ec2-metadata-token: %s\n' "$imds_token" > "$D/imds-header"
    code=$(curl --noproxy '*' --connect-timeout 1 --max-time 2 -sS -o "$D/iam-info" -w '%{http_code}' -H "@$D/imds-header" http://169.254.169.254/latest/meta-data/iam/info 2>/dev/null || true)
    rm -f -- "$D/imds-header"
    emit aws_imds_iam_info "$([ "$code" = 200 ] && grep -q 'InstanceProfileArn' "$D/iam-info" && echo true || echo false)"
  else
    emit aws_imds_iam_info false
  fi
  unset imds_token
  code=$(curl --noproxy '*' --connect-timeout 1 --max-time 2 -sS -o "$D/gcp-scopes" -w '%{http_code}' -H 'Metadata-Flavor: Google' http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/scopes 2>/dev/null || true)
  emit gcp_metadata_scopes "$([ "$code" = 200 ] && echo true || echo false)"
  if [ "$code" = 200 ]; then
    emit gcp_cloud_platform_scope "$(grep -q 'https://www.googleapis.com/auth/cloud-platform' "$D/gcp-scopes" && echo true || echo false)"
    emit gcp_storage_write_scope "$(grep -Eq 'https://www.googleapis.com/auth/devstorage\.(read_write|full_control)' "$D/gcp-scopes" && echo true || echo false)"
  fi
fi
emit completed true
