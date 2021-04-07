# -*- coding: utf-8 -*-
import unittest
from tools import print_test_stats


def fakehash(char):
    return char * 40


def dummy_meta_meta() -> print_test_stats.ReportMetaMeta:
    return {
        'build_pr': '',
        'build_tag': '',
        'build_sha1': '',
        'build_branch': '',
        'build_job': '',
        'build_workflow_id': '',
    }


def makecase(name, seconds, *, errored=False, failed=False, skipped=False):
    return {
        'name': name,
        'seconds': seconds,
        'errored': errored,
        'failed': failed,
        'skipped': skipped,
    }


def make_report_v1(tests) -> print_test_stats.Version1Report:
    suites = {
        suite_name: {
            'total_seconds': sum(case['seconds'] for case in cases),
            'cases': cases,
        }
        for suite_name, cases in tests.items()
    }
    return {
        **dummy_meta_meta(),
        'total_seconds': sum(s['total_seconds'] for s in suites.values()),
        'suites': suites,
    }


def make_case_v2(seconds, status=None) -> print_test_stats.Version2Case:
    return {
        'seconds': seconds,
        'status': status,
    }


def make_report_v2(tests) -> print_test_stats.Version2Report:
    files = {}
    for file_name, file_suites in tests.items():
        suites = {
            suite_name: {
                'total_seconds': sum(case['seconds'] for case in cases.values()),
                'cases': cases,
            }
            for suite_name, cases in file_suites.items()
        }
        files[file_name] = {
            'suites': suites,
            'total_seconds': sum(suite['total_seconds'] for suite in suites.values()),
        }
    return {
        **dummy_meta_meta(),
        'format_version': 2,
        'total_seconds': sum(s['total_seconds'] for s in files.values()),
        'files': files,
    }
maxDiff = None

class TestPrintTestStats(unittest.TestCase):
    version1_report: print_test_stats.Version1Report = make_report_v1({
        # input ordering of the suites is ignored
        'Grault': [
            # not printed: status same and time similar
            makecase('test_grault0', 4.78, failed=True),
            # status same, but time increased a lot
            makecase('test_grault2', 1.473, errored=True),
        ],
        # individual tests times changed, not overall suite
        'Qux': [
            # input ordering of the test cases is ignored
            makecase('test_qux1', 0.001, skipped=True),
            makecase('test_qux6', 0.002, skipped=True),
            # time in bounds, but status changed
            makecase('test_qux4', 7.158, failed=True),
            # not printed because it's the same as before
            makecase('test_qux7', 0.003, skipped=True),
            makecase('test_qux5', 11.968),
            makecase('test_qux3', 23.496),
        ],
        # new test suite
        'Bar': [
            makecase('test_bar2', 3.742, failed=True),
            makecase('test_bar1', 50.447),
        ],
        # overall suite time changed but no individual tests
        'Norf': [
            makecase('test_norf1', 3),
            makecase('test_norf2', 3),
            makecase('test_norf3', 3),
            makecase('test_norf4', 3),
        ],
        # suite doesn't show up if it doesn't change enough
        'Foo': [
            makecase('test_foo1', 42),
            makecase('test_foo2', 56),
        ],
    })

    version2_report: print_test_stats.Version2Report = make_report_v2(
        {
            'test_a': {
                'Grault': {
                    'test_grault0': make_case_v2(4.78, 'failed'),
                    'test_grault2': make_case_v2(1.473, 'errored'),
                },
                'Qux': {
                    'test_qux1': make_case_v2(0.001, 'skipped'),
                    'test_qux6': make_case_v2(0.002, 'skipped'),
                    'test_qux4': make_case_v2(7.158, 'failed'),
                    'test_qux7': make_case_v2(0.003, 'skipped'),
                    'test_qux8': make_case_v2(11.968),
                    'test_qux3': make_case_v2(23.496),
                }
            },
            'test_b': {
                'Bar': {
                    'test_bar2': make_case_v2(3.742, 'failed'),
                    'test_bar1': make_case_v2(50.447),
                },
                # overall suite time changed but no individual tests
                'Norf': {
                    'test_norf1': make_case_v2(3),
                    'test_norf2': make_case_v2(3),
                    'test_norf3': make_case_v2(3),
                    'test_norf4': make_case_v2(3),
                },
            },
            'test_c': {
                'Foo': {
                    'test_foo1': make_case_v2(42),
                    'test_foo2': make_case_v2(56),
                },
            }
        })

    def test_simplify(self):
        self.assertEqual(
            {
                '': {
                    'Bar': {
                        'test_bar1': {'seconds': 50.447, 'status': None},
                        'test_bar2': {'seconds': 3.742, 'status': 'failed'},
                    },
                    'Foo': {
                        'test_foo1': {'seconds': 42, 'status': None},
                        'test_foo2': {'seconds': 56, 'status': None},
                    },
                    'Grault': {
                        'test_grault0': {'seconds': 4.78, 'status': 'failed'},
                        'test_grault2': {'seconds': 1.473, 'status': 'errored'},
                    },
                    'Norf': {
                        'test_norf1': {'seconds': 3, 'status': None},
                        'test_norf3': {'seconds': 3, 'status': None},
                        'test_norf2': {'seconds': 3, 'status': None},
                        'test_norf4': {'seconds': 3, 'status': None},
                    },
                    'Qux': {
                        'test_qux1': {'seconds': 0.001, 'status': 'skipped'},
                        'test_qux3': {'seconds': 23.496, 'status': None},
                        'test_qux4': {'seconds': 7.158, 'status': 'failed'},
                        'test_qux5': {'seconds': 11.968, 'status': None},
                        'test_qux6': {'seconds': 0.002, 'status': 'skipped'},
                        'test_qux7': {'seconds': 0.003, 'status': 'skipped'},
                    },
                },
            },
            print_test_stats.simplify(self.version1_report)
        )

        self.assertEqual(
            {
                'test_a': {
                    'Grault': {
                        'test_grault0': {'seconds': 4.78, 'status': 'failed'},
                        'test_grault2': {'seconds': 1.473, 'status': 'errored'},
                    },
                    'Qux': {
                        'test_qux1': {'seconds': 0.001, 'status': 'skipped'},
                        'test_qux3': {'seconds': 23.496, 'status': None},
                        'test_qux4': {'seconds': 7.158, 'status': 'failed'},
                        'test_qux6': {'seconds': 0.002, 'status': 'skipped'},
                        'test_qux7': {'seconds': 0.003, 'status': 'skipped'},
                        'test_qux8': {'seconds': 11.968, 'status': None},
                    },
                },
                'test_b': {
                    'Bar': {
                        'test_bar1': {'seconds': 50.447, 'status': None},
                        'test_bar2': {'seconds': 3.742, 'status': 'failed'},
                    },
                    'Norf': {
                        'test_norf1': {'seconds': 3, 'status': None},
                        'test_norf2': {'seconds': 3, 'status': None},
                        'test_norf3': {'seconds': 3, 'status': None},
                        'test_norf4': {'seconds': 3, 'status': None},
                    },
                },
                'test_c': {
                    'Foo': {
                        'test_foo1': {'seconds': 42, 'status': None},
                        'test_foo2': {'seconds': 56, 'status': None},
                    },
                },
            },
            print_test_stats.simplify(self.version2_report),
        )

    def test_analysis(self):
        head_report = self.version1_report

        base_reports = {
            # bbbb has no reports, so base is cccc instead
            fakehash('b'): [],
            fakehash('c'): [
                make_report_v1({
                    'Baz': [
                        makecase('test_baz2', 13.605),
                        # no recent suites have & skip this test
                        makecase('test_baz1', 0.004, skipped=True),
                    ],
                    'Foo': [
                        makecase('test_foo1', 43),
                        # test added since dddd
                        makecase('test_foo2', 57),
                    ],
                    'Grault': [
                        makecase('test_grault0', 4.88, failed=True),
                        makecase('test_grault1', 11.967, failed=True),
                        makecase('test_grault2', 0.395, errored=True),
                        makecase('test_grault3', 30.460),
                    ],
                    'Norf': [
                        makecase('test_norf1', 2),
                        makecase('test_norf2', 2),
                        makecase('test_norf3', 2),
                        makecase('test_norf4', 2),
                    ],
                    'Qux': [
                        makecase('test_qux3', 4.978, errored=True),
                        makecase('test_qux7', 0.002, skipped=True),
                        makecase('test_qux2', 5.618),
                        makecase('test_qux4', 7.766, errored=True),
                        makecase('test_qux6', 23.589, failed=True),
                    ],
                }),
            ],
            fakehash('d'): [
                make_report_v1({
                    'Foo': [
                        makecase('test_foo1', 40),
                        # removed in cccc
                        makecase('test_foo3', 17),
                    ],
                    'Baz': [
                        # not skipped, so not included in stdev
                        makecase('test_baz1', 3.14),
                    ],
                    'Qux': [
                        makecase('test_qux7', 0.004, skipped=True),
                        makecase('test_qux2', 6.02),
                        makecase('test_qux4', 20.932),
                    ],
                    'Norf': [
                        makecase('test_norf1', 3),
                        makecase('test_norf2', 3),
                        makecase('test_norf3', 3),
                        makecase('test_norf4', 3),
                    ],
                    'Grault': [
                        makecase('test_grault0', 5, failed=True),
                        makecase('test_grault1', 14.325, failed=True),
                        makecase('test_grault2', 0.31, errored=True),
                    ],
                }),
            ],
            fakehash('e'): [],
            fakehash('f'): [
                make_report_v1({
                    'Foo': [
                        makecase('test_foo3', 24),
                        makecase('test_foo1', 43),
                    ],
                    'Baz': [
                        makecase('test_baz2', 16.857),
                    ],
                    'Qux': [
                        makecase('test_qux2', 6.422),
                        makecase('test_qux4', 6.382, errored=True),
                    ],
                    'Norf': [
                        makecase('test_norf1', 0.9),
                        makecase('test_norf3', 0.9),
                        makecase('test_norf2', 0.9),
                        makecase('test_norf4', 0.9),
                    ],
                    'Grault': [
                        makecase('test_grault0', 4.7, failed=True),
                        makecase('test_grault1', 13.146, failed=True),
                        makecase('test_grault2', 0.48, errored=True),
                    ],
                }),
            ],
        }

        simpler_head = print_test_stats.simplify(head_report)
        simpler_base = {}
        for commit, reports in base_reports.items():
            simpler_base[commit] = [print_test_stats.simplify(r) for r in reports]
        analysis = print_test_stats.analyze(
            head_report=simpler_head,
            base_reports=simpler_base,
        )

        self.assertEqual(
            '''\

- class Baz:
-     # was   15.23s ±   2.30s
-
-     def test_baz1: ...
-         # was   0.004s           (skipped)
-
-     def test_baz2: ...
-         # was  15.231s ±  2.300s


  class Grault:
      # was   48.86s ±   1.19s
      # now    6.25s

    - def test_grault1: ...
    -     # was  13.146s ±  1.179s (failed)

    - def test_grault3: ...
    -     # was  30.460s


  class Qux:
      # was   41.66s ±   1.06s
      # now   42.63s

    - def test_qux2: ...
    -     # was   6.020s ±  0.402s

    ! def test_qux3: ...
    !     # was   4.978s           (errored)
    !     # now  23.496s

    ! def test_qux4: ...
    !     # was   7.074s ±  0.979s (errored)
    !     # now   7.158s           (failed)

    ! def test_qux6: ...
    !     # was  23.589s           (failed)
    !     # now   0.002s           (skipped)

    + def test_qux1: ...
    +     # now   0.001s           (skipped)

    + def test_qux5: ...
    +     # now  11.968s


+ class Bar:
+     # now   54.19s
+
+     def test_bar1: ...
+         # now  50.447s
+
+     def test_bar2: ...
+         # now   3.742s           (failed)

''',
            print_test_stats.anomalies(analysis),
        )

    def test_graph(self):
        # HEAD is on master
        self.assertEqual(
            '''\
Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    * aaaaaaaaaa (HEAD)              total time   502.99s
    * bbbbbbbbbb (base)   1 report,  total time    47.84s
    * cccccccccc          1 report,  total time   332.50s
    * dddddddddd          0 reports
    |
    :
''',
            print_test_stats.graph(
                head_sha=fakehash('a'),
                head_seconds=502.99,
                base_seconds={
                    fakehash('b'): [47.84],
                    fakehash('c'): [332.50],
                    fakehash('d'): [],
                },
                on_master=True,
            )
        )

        self.assertEqual(
            '''\
Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    | * aaaaaaaaaa (HEAD)            total time  9988.77s
    |/
    * bbbbbbbbbb (base) 121 reports, total time  7654.32s ±   55.55s
    * cccccccccc         20 reports, total time  5555.55s ±  253.19s
    * dddddddddd          1 report,  total time  1234.56s
    |
    :
''',
            print_test_stats.graph(
                head_sha=fakehash('a'),
                head_seconds=9988.77,
                base_seconds={
                    fakehash('b'): [7598.77] * 60 + [7654.32] + [7709.87] * 60,
                    fakehash('c'): [5308.77] * 10 + [5802.33] * 10,
                    fakehash('d'): [1234.56],
                },
                on_master=False,
            )
        )

        self.assertEqual(
            '''\
Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    | * aaaaaaaaaa (HEAD)            total time    25.52s
    | |
    | : (5 commits)
    |/
    * bbbbbbbbbb          0 reports
    * cccccccccc          0 reports
    * dddddddddd (base)  15 reports, total time    58.92s ±   25.82s
    |
    :
''',
            print_test_stats.graph(
                head_sha=fakehash('a'),
                head_seconds=25.52,
                base_seconds={
                    fakehash('b'): [],
                    fakehash('c'): [],
                    fakehash('d'): [52.25] * 14 + [152.26],
                },
                on_master=False,
                ancestry_path=5,
            )
        )

        self.assertEqual(
            '''\
Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    | * aaaaaaaaaa (HEAD)            total time     0.08s
    |/|
    | : (1 commit)
    |
    * bbbbbbbbbb          0 reports
    * cccccccccc (base)   1 report,  total time     0.09s
    * dddddddddd          3 reports, total time     0.10s ±    0.05s
    |
    :
''',
            print_test_stats.graph(
                head_sha=fakehash('a'),
                head_seconds=0.08,
                base_seconds={
                    fakehash('b'): [],
                    fakehash('c'): [0.09],
                    fakehash('d'): [0.05, 0.10, 0.15],
                },
                on_master=False,
                other_ancestors=1,
            )
        )

        self.assertEqual(
            '''\
Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    | * aaaaaaaaaa (HEAD)            total time     5.98s
    | |
    | : (1 commit)
    |/|
    | : (7 commits)
    |
    * bbbbbbbbbb (base)   2 reports, total time     6.02s ±    1.71s
    * cccccccccc          0 reports
    * dddddddddd         10 reports, total time     5.84s ±    0.92s
    |
    :
''',
            print_test_stats.graph(
                head_sha=fakehash('a'),
                head_seconds=5.98,
                base_seconds={
                    fakehash('b'): [4.81, 7.23],
                    fakehash('c'): [],
                    fakehash('d'): [4.97] * 5 + [6.71] * 5,
                },
                on_master=False,
                ancestry_path=1,
                other_ancestors=7,
            )
        )

    def test_regression_info(self):
        self.assertEqual(
            '''\
----- Historic stats comparison result ------

    job: foo_job
    commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    | * aaaaaaaaaa (HEAD)            total time     3.02s
    |/
    * bbbbbbbbbb (base)   1 report,  total time    41.00s
    * cccccccccc          1 report,  total time    43.00s
    |
    :

Removed  (across    1 suite)      1 test,  totaling -   1.00s
Modified (across    1 suite)      1 test,  totaling -  41.48s ±   2.12s
Added    (across    1 suite)      1 test,  totaling +   3.00s
''',
            print_test_stats.regression_info(
                head_sha=fakehash('a'),
                head_report=make_report_v1({
                    'Foo': [
                        makecase('test_foo', 0.02, skipped=True),
                        makecase('test_baz', 3),
                    ]}),
                base_reports={
                    fakehash('b'): [
                        make_report_v1({
                            'Foo': [
                                makecase('test_foo', 40),
                                makecase('test_bar', 1),
                            ],
                        }),
                    ],
                    fakehash('c'): [
                        make_report_v1({
                            'Foo': [
                                makecase('test_foo', 43),
                            ],
                        }),
                    ],
                },
                job_name='foo_job',
                on_master=False,
                ancestry_path=0,
                other_ancestors=0,
            )
        )

    def test_regression_info_new_job(self):
        self.assertEqual(
            '''\
----- Historic stats comparison result ------

    job: foo_job
    commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    | * aaaaaaaaaa (HEAD)            total time     3.02s
    | |
    | : (3 commits)
    |/|
    | : (2 commits)
    |
    * bbbbbbbbbb          0 reports
    * cccccccccc          0 reports
    |
    :

Removed  (across    0 suites)     0 tests, totaling     0.00s
Modified (across    0 suites)     0 tests, totaling     0.00s
Added    (across    1 suite)      2 tests, totaling +   3.02s
''',
            print_test_stats.regression_info(
                head_sha=fakehash('a'),
                head_report=make_report_v1({
                    'Foo': [
                        makecase('test_foo', 0.02, skipped=True),
                        makecase('test_baz', 3),
                    ]}),
                base_reports={
                    fakehash('b'): [],
                    fakehash('c'): [],
                },
                job_name='foo_job',
                on_master=False,
                ancestry_path=3,
                other_ancestors=2,
            )
        )


if __name__ == '__main__':
    unittest.main()
