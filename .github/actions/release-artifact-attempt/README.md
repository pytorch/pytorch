# Ordinary artifact attempt audit

Never merge this audit-only change. PR0, workflow ID0 and attempt0 are inert.
Activation requires a separately reviewed exact-owned draft, normal workflow
approval, and a fresh check that the unique workflow name and artifact prefix
cannot enter an automatic publisher. Do not use a production statistics prefix.

The controller binds producer attempt `N` before activation. Approval can make
the first executed job attempt2; this is not assumed to be attempt1. `N` first
proves genuine same-job deletion, then retains one deterministic, ordinary
`canary.txt` ZIP. It emits only its run/head/job/attempt and artifact identity.
The controller authenticates that identity through GitHub, stores the immutable
receipt locally, then reruns only that exact run and head.

`N+1` has no experimental uploader. It independently requires the same run,
head, workflow and one exact artifact name with the precomputed size/digest.
The unchanged pinned PyTorch helper runs with its scoped job token in a private
network namespace. A CONNECT relay admits only `api.github.com` and the exact
public HTTPS host returned for that owned artifact. A Python profiling hook
observes the original helper's numeric artifact-download URL; it does not alter
the target source, HTTP implementation or response. Metadata is revalidated
after downloading. The controller must compare the emitted ID against its
pre-rerun receipt before accepting the result. Same bytes are not enough when
the IDs disagree.

The relay deliberately supports only IPv4 answers outside its numeric
special-use CIDR denylist. An IPv6-only answer fails closed; there is no IPv6
fallback. The denied-host control requires an admission-denied observation with
zero DNS lookups and zero dials, not merely a failed request.

The current `torchci-reports` opposing control is local-only and must return
without entering the original extraction or S3-upload functions. The profiling
guard records and blocks either entry; the control rejects even a caught error.
No stats uploader, Dr.CI endpoint, AWS method,
publication workflow, OIDC role or persistent cache is invoked. This proves
only the owned same-run helper behavior, not protected publication impact.

The controller retains deletion authority for every acknowledged artifact ID.
After consumption or failure it rechecks the exact ownership/name/digest,
deletes only those artifacts, verifies HTTP204 followed by GET404 and an empty
owned-run listing, and closes the draft unmerged. The job deletes incomplete
same-job production when possible; it deliberately retains successful producer
artifacts for the same-run rerun and independent controller cleanup.

Local tests use explicit unit service fixtures only for protocol/identity
failure states. The real network-namespace and unchanged Python opposing
controls use no external service. Provision a managed CPython3.13.13 venv at
`agent_space/consumer-venv` with the hash-locked `requirements.txt`, then run
`node --test .github/actions/release-artifact-attempt/index.test.cjs`. The tests
also need the repository's managed build/lint `.venv` and a genuine
download-artifact checkout at `agent_space/artifact-attempt-sdk-source`, pinned
to `d3f86a106a0bac45b974a628896c90dbdf5c8093`. A sparse checkout containing
`action.yml`, `package.json` and `package-lock.json` is sufficient.
