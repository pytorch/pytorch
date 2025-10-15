## How to update moodycamel

To update the moodycamel directory with the latest files from the concurrentqueue repository, run the following command:

```bash
cd third_party/concurrentqueue
./update.sh
```

## Why not a submodule

We didn’t want to deal with license issues from the test/ directory so we decided on a non-submodule approach. 
This script allows us to keep the moodycamel directory up-to-date with the latest files from the concurrentqueue 
repository without having to worry about submodule complexities.

## Original source

repo: https://github.com/cameron314/concurrentqueue
commit: 24b78782bd6ca5a5853ef46917708806112dc142
