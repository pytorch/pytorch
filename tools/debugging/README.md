[`symbolize_from_artifact.py`](symbolize_from_artifact.py) is a helper script for reading crash dumps from CI runs. On an unstructured exception (e.g. a segfault), GitHub Actions Linux tests will upload a [minidump](https://chromium.googlesource.com/breakpad/breakpad/+/master/docs/getting_started_with_breakpad.md) of the crash via the [breakpad](https://github.com/google/breakpad) library. From this crash you can get a stack trace of the failure by following these steps:

1. Go to the HUD page for your PR at `https://hud.pytorch.org/pr/<PR number>`

2. Download the `[gha] crash-reports-default` file from the failing test job and unzip it

    ```bash
    # Download 'crash-reports-default' from the CI job artifacts on GitHub and
    # (if necessary) copy it to your Linux remote server via scp
    unzip crash-reports-default.zip
    ```

2. Download the build artifacts named `[s3] artifacts.zip` and unzip it

    ```bash
    # Download 'crash-reports-default' from the CI job artifacts on GitHub and
    # (if necessary) copy it to your Linux remote server via scp
    unzip artifacts.zip
    ```

3. Use the helper script and view the stack trace in the output (it may help to search for `(crashed)` to find the faulting thread.

    ```bash
    python symbolize_from_artifact.py --pytorch-build-dir build  --minidump abc-def-123-456.dmp
    ```