import json
import os
import re
import unittest

from job_name_match import choose_test_job_family


def job(name, job_id):
    return {"name": name, "id": job_id}


class ChooseTestJobFamilyTest(unittest.TestCase):
    def test_rx7900_replaces_navi31_topology(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "parity_job_config.json"
        )
        with open(config_path) as config_file:
            rocm = json.load(config_file)["rocm"]

        self.assertNotIn("navi31", rocm)
        self.assertEqual(
            rocm["rx7900"],
            {
                "default": [{
                    "workflow": "rocm-rx7900",
                    "job_prefix": "linux-jammy-rocm-py3.11-rx7900",
                }],
                "shard_counts": {"default": 2},
                "checkrun_regex": "rocm.*rx7900.*/ test [(]default,",
            },
        )

    def test_cuda_inductor_source_is_trunk(self):
        """pytorch/pytorch#199291 moved the only CUDA inductor config to trunk."""
        config_path = os.path.join(
            os.path.dirname(__file__), "parity_job_config.json"
        )
        with open(config_path) as config_file:
            cuda = json.load(config_file)["cuda"]

        self.assertEqual(
            cuda["inductor"],
            [{
                "workflow": "trunk",
                "job_prefix": "linux-jammy-cuda13.2-py3.11-gcc11",
            }],
        )

        regex = re.compile(cuda["checkrun_regex"])
        for config in ("default", "distributed", "inductor"):
            with self.subTest(config=config):
                self.assertRegex(
                    "linux-jammy-cuda13.2-py3.11-gcc11 / test "
                    f"({config}, 1, 2, mt-l-x86aavx2-11-41-l4)",
                    regex,
                )
        for name in (
            # Retired by #199291; inductor-unittest.yml has no inductor config.
            "unit-test / inductor-test-cuda132 / test (inductor, 1, 2, runner)",
            "linux-noble-rocm-py3.11-mi350 / test (inductor, 1, 2, gfx950)",
            "linux-jammy-cuda13.2-py3.11-gcc11 / test "
            "(inductor_cpp_wrapper, 1, 2, a10g)",
        ):
            with self.subTest(name=name):
                self.assertNotRegex(name, regex)

    def test_cuda_inductor_family_resolves_in_a_shared_trunk_run(self):
        """One trunk run now holds both platforms' inductor jobs."""
        jobs = [
            job(
                "linux-jammy-cuda13.2-py3.11-gcc11 / test "
                f"(inductor, {shard}, 2, l4)",
                100 + shard,
            )
            for shard in (1, 2)
        ] + [
            job(
                "linux-noble-rocm-py3.11-mi350 / test "
                f"(inductor, {shard}, 2, gfx950)",
                200 + shard,
            )
            for shard in (1, 2)
        ]

        family = choose_test_job_family(
            jobs, "inductor", "cuda", "linux-jammy-cuda13.2-py3.11-gcc11"
        )
        self.assertEqual(family["prefix"], "linux-jammy-cuda13.2-py3.11-gcc11")
        self.assertEqual(family["total"], 2)
        self.assertTrue(family["complete"])
        self.assertEqual(
            sorted(j["id"] for j in family["jobs"]), [101, 102]
        )

    def test_discovers_renamed_cuda_family(self):
        jobs = [
            job(
                f"linux-jammy-cuda13.2-py3.10-gcc11 / test "
                f"(default, {shard}, 14, runner)",
                shard,
            )
            for shard in range(1, 15)
        ]
        family = choose_test_job_family(
            jobs, "default", "cuda", "linux-jammy-cuda13.0-py3.10-gcc11"
        )
        self.assertEqual(family["prefix"], "linux-jammy-cuda13.2-py3.10-gcc11")
        self.assertEqual(family["total"], 14)

    def test_prefers_normal_cuda_family_over_debug(self):
        jobs = [
            job(
                f"linux-jammy-cuda13.2-py3.10-gcc11 / test "
                f"(default, {shard}, 14, runner)",
                shard,
            )
            for shard in range(1, 15)
        ]
        jobs.extend(
            job(
                f"linux-jammy-cuda13.0-py3.10-gcc11-debug / test "
                f"(default, {shard}, 7, runner)",
                100 + shard,
            )
            for shard in range(1, 8)
        )
        family = choose_test_job_family(
            jobs, "default", "cuda", "linux-jammy-cuda13.0-py3.10-gcc11"
        )
        self.assertEqual(family["prefix"], "linux-jammy-cuda13.2-py3.10-gcc11")

    def test_prefers_complete_renamed_family_over_incomplete_exact_match(self):
        jobs = [
            job(
                "linux-jammy-cuda13.0-py3.10-gcc11 / test "
                "(default, 1, 14, runner)",
                1,
            )
        ]
        jobs.extend(
            job(
                f"linux-jammy-cuda13.2-py3.10-gcc11 / test "
                f"(default, {shard}, 14, runner)",
                100 + shard,
            )
            for shard in range(1, 15)
        )
        family = choose_test_job_family(
            jobs, "default", "cuda", "linux-jammy-cuda13.0-py3.10-gcc11"
        )
        self.assertEqual(family["prefix"], "linux-jammy-cuda13.2-py3.10-gcc11")

    def test_discovers_historical_rocm_family(self):
        jobs = [
            job(
                f"linux-jammy-rocm-py3.10-mi350 / test "
                f"(distributed, {shard}, 3, runner)",
                shard,
            )
            for shard in range(1, 4)
        ]
        family = choose_test_job_family(
            jobs, "distributed", "rocm", "linux-noble-rocm-py3.11-mi350"
        )
        self.assertEqual(family["prefix"], "linux-jammy-rocm-py3.10-mi350")
        self.assertEqual(family["total"], 3)

    def test_preserves_discovered_test_kind(self):
        jobs = [
            job(
                f"linux-jammy-cuda13.2-py3.10-gcc11 / test-osdc "
                f"(distributed, {shard}, 10, runner)",
                shard,
            )
            for shard in range(1, 11)
        ]
        family = choose_test_job_family(
            jobs, "distributed", "cuda", "linux-jammy-cuda13.2-py3.10-gcc11"
        )
        self.assertEqual(family["kind"], "test-osdc")

    def test_marks_missing_shards_incomplete(self):
        jobs = [
            job(
                "linux-jammy-cuda13.2-py3.10-gcc11 / test "
                "(default, 1, 14, runner)",
                1,
            )
        ]
        family = choose_test_job_family(
            jobs, "default", "cuda", "linux-jammy-cuda13.2-py3.10-gcc11"
        )
        self.assertFalse(family["complete"])

    def test_keeps_platforms_separate(self):
        jobs = [
            job(
                "linux-jammy-cuda13.2-py3.10-gcc11 / test "
                "(default, 1, 14, runner)",
                1,
            )
        ]
        self.assertIsNone(
            choose_test_job_family(
                jobs, "default", "rocm", "linux-noble-rocm-py3.11-mi350"
            )
        )


if __name__ == "__main__":
    unittest.main()
