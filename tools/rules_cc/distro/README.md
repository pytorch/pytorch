# Package rules_cc

```bash
bazel build :relnotes
cat ../bazel-bin/distro/relnotes.txt
tar tzf ../bazel-bin/distro/rules_cc-*.tar.gz
```

- Create a new release
- Copy/paste relnotes.txt into the notes. Adjust as needed.
- Upload the tar.gz file as an artifact.
