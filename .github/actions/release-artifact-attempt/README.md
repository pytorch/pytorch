# Ordinary artifact attempt audit

Never merge this audit-only change. The reviewed inert predecessor allocated
draft PR205523 and workflow422814113. This append-only activation binds producer
attempt2 and consumer attempt3. The producer succeeds; a separate consumer
job fails only at its fixed hold step before a specific-job rerun. It still requires independent final-tree review,
normal exact-head workflow approval, and a fresh check that the unique workflow
name and artifact prefix cannot enter an automatic publisher. Do not use a
production statistics prefix.

The controller binds producer attempt `N` before activation. Approval can make
the first executed job attempt2; this is not assumed to be attempt1. `N` first
proves genuine same-job deletion, then retains one deterministic, ordinary
`canary.txt` ZIP. It emits only its run/head/job/attempt and artifact identity.
The controller authenticates that identity through GitHub, stores the immutable
receipt locally, then reruns only the authenticated numeric consumer job ID. Never use an
all-jobs rerun. The controller retains the full authenticated source jobs and
their canonical fingerprint. Excluded prerequisites may retain their original
records, or use the narrowly authenticated GitHub copied-record shape: new ID,
current attempt, empty steps, null runner-group ID, and unchanged original
execution metadata. Their original completion must precede the current run
start, which must precede creation of the copy. Any new producer execution or
changed source evidence fails closed. The exported `rerunTarget` selector only
describes that numeric job endpoint; it never sends a request.

`N` consumer hold has no SDK client, archive operation, helper invocation or
fixture directory. Only its fixed named hold step may fail. `N+1` has no
experimental uploader. It independently requires the same run,
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

The selected merge must have the reviewed head parent and identical workflow,
check-CI and entire action-tree Git objects. Runtime guards independently
reauthenticate that source before an operation. Missing or different service
metadata fails closed; unit fixtures are not evidence of partial-rerun retention.

The controller retains deletion authority for every acknowledged artifact ID.
After consumption or failure it rechecks the exact ownership/name/digest,
deletes only those still present, verifies HTTP204 followed by GET404 and an
empty owned-run listing, and closes the draft unmerged. If an artifact was
already absent, record its404 without claiming a controller DELETE204. The job deletes incomplete
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
