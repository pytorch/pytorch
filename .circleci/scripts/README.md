All the scripts in this directory are callable from `~/workspace/.circleci/scripts/foo.sh`.
Don't try to call them as `.circleci/scripts/foo.sh`, that won't
(necessarily) work.  See Note [Workspace for CircleCI scripts] in
job-specs-setup.yml for more details.
