-- One row per test file per change of its # Owner(s): header, append-only.
-- See proposal.md section 4.2.

CREATE TABLE tests.owners
(
    -- What: repository the file lives in, e.g. pytorch/pytorch.
    -- Derived: GITHUB_REPOSITORY of the job that parses the headers, in the same
    --   form as tests.tests.repo.
    -- Used: with file, the join key to tests.tests; leading sort key.
    repo       LowCardinality(String),

    -- What: repo-relative path of the test file, e.g. test/test_torch.py.
    -- Derived: the path of the file whose header was read, in the same form as
    --   tests.tests.file.
    -- Used: with repo, the join key to tests.tests; second sort key.
    file       LowCardinality(String),

    -- What: the owner labels in the file's header, e.g. ['module: nn']; empty once
    --   the file is deleted.
    -- Derived: the # Owner(s): line, parsed as the TESTOWNERS linter does.
    -- Used: owner grouping and routing.
    owners     Array(LowCardinality(String)),

    -- What: when the job first saw this header value on main.
    -- Derived: clock of the job that parses the headers on main; it appends a row
    --   only when a file's owners differ from that file's newest row.
    -- Used: current owner (newest row per repo and file) and owner at a past time
    --   (newest row at or before it).
    created_at DateTime
)
ENGINE = MergeTree
ORDER BY (repo, file, created_at);
