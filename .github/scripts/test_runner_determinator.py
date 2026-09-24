import tempfile
from argparse import Namespace
from pathlib import Path
from unittest import main, TestCase
from unittest.mock import Mock, patch

import runner_determinator as rd


USER_BRANCH = "somebranch"
EXCEPTION_BRANCH = "main"


class TestRunnerDeterminatorIssueParser(TestCase):
    def test_parse_settings(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        settings = rd.parse_settings(settings_text)

        self.assertTupleEqual(
            rd.Experiment(rollout_perc=25),
            settings.experiments["lf"],
            "lf settings not parsed correctly",
        )
        self.assertTupleEqual(
            rd.Experiment(rollout_perc=0, default=False),
            settings.experiments["otherExp"],
            "otherExp settings not parsed correctly",
        )

    def test_parse_settings_with_invalid_experiment_name_skips_experiment(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            -badExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,-badExp

        """

        settings = rd.parse_settings(settings_text)

        self.assertTupleEqual(
            rd.Experiment(rollout_perc=25),
            settings.experiments["lf"],
            "lf settings not parsed correctly",
        )
        self.assertNotIn("-badExp", settings.experiments)

    def test_parse_settings_in_code_block(self) -> None:
        settings_text = """

        ```
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 0
                default: false
        ```

        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        settings = rd.parse_settings(settings_text)

        self.assertTupleEqual(
            rd.Experiment(rollout_perc=25),
            settings.experiments["lf"],
            "lf settings not parsed correctly",
        )
        self.assertTupleEqual(
            rd.Experiment(rollout_perc=0, default=False),
            settings.experiments["otherExp"],
            "otherExp settings not parsed correctly",
        )

    def test_parse_all_branches_setting(self) -> None:
        settings_text = """
        ```
        experiments:
            lf:
                rollout_perc: 25
                all_branches: true
            otherExp:
                all_branches: True
                rollout_perc: 0
        ```

        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        settings = rd.parse_settings(settings_text)

        self.assertTupleEqual(
            rd.Experiment(rollout_perc=25, all_branches=True),
            settings.experiments["lf"],
            "lf settings not parsed correctly",
        )
        self.assertTrue(settings.experiments["otherExp"].all_branches)
        self.assertTupleEqual(
            rd.Experiment(rollout_perc=0, all_branches=True),
            settings.experiments["otherExp"],
            "otherExp settings not parsed correctly",
        )

    def test_parse_users(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [rd.UserExperimentConfig("lf", 100)],
            users["User1"],
        )
        self.assertEqual(
            [
                rd.UserExperimentConfig("lf", 100),
                rd.UserExperimentConfig("otherExp", 100),
            ],
            users["User2"],
        )

    def test_parse_users_without_settings(self) -> None:
        settings_text = """

        @User1,lf
        @User2,lf,otherExp

        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [rd.UserExperimentConfig("lf", 100)],
            users["User1"],
        )
        self.assertEqual(
            [
                rd.UserExperimentConfig("lf", 100),
                rd.UserExperimentConfig("otherExp", 100),
            ],
            users["User2"],
        )

    def test_parse_users_with_rollout_perc(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            arc:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,arc:10
        @User2,arc:50
        @User3,lf

        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [
                rd.UserExperimentConfig("lf", 100),
                rd.UserExperimentConfig("arc", 10),
            ],
            users["User1"],
        )
        self.assertEqual(
            [rd.UserExperimentConfig("arc", 50)],
            users["User2"],
        )
        self.assertEqual(
            [rd.UserExperimentConfig("lf", 100)],
            users["User3"],
        )

    def test_parse_users_invalid_percentage_defaults_to_100(self) -> None:
        """Non-numeric percentage like arc:abc should default to 100%."""
        settings_text = """
        @User1,arc:abc
        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [rd.UserExperimentConfig("arc", 100)],
            users["User1"],
        )

    def test_parse_users_negative_percentage_clamped_to_zero(self) -> None:
        """Negative percentage like arc:-5 should be clamped to 0."""
        settings_text = """
        @User1,arc:-5
        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [rd.UserExperimentConfig("arc", 0)],
            users["User1"],
        )

    def test_parse_users_over_100_percentage_clamped(self) -> None:
        """Percentage over 100 like arc:200 should be clamped to 100."""
        settings_text = """
        @User1,arc:200
        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [rd.UserExperimentConfig("arc", 100)],
            users["User1"],
        )

    def test_parse_users_opt_out_ignores_percentage(self) -> None:
        """Opt-out entries like -lf should not parse a percentage."""
        settings_text = """
        @User1,-lf
        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [rd.UserExperimentConfig("-lf", 100)],
            users["User1"],
        )


class TestRunnerDeterminatorGetRunnerPrefix(TestCase):
    def test_opted_in_user(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for User1")

    def test_explicitly_opted_out_user(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,-lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for User1")

    def test_explicitly_opted_in_and_out_user_should_opt_out(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,-lf,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for User1")

    def test_opted_in_user_two_experiments(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(settings_text, ["User2"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for User2")

    def test_opted_in_user_two_experiments_default(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(settings_text, ["User2"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for User2")

    def test_opted_in_user_two_experiments_default_exp(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(
            settings_text, ["User2"], USER_BRANCH, frozenset(["lf", "otherExp"])
        )
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for User2")

    def test_opted_in_user_two_experiments_default_exp_2(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(
            settings_text, ["User2"], USER_BRANCH, frozenset(["otherExp"])
        )
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for User2")

    @patch("random.uniform", return_value=50)
    def test_opted_out_user(self, mock_uniform: Mock) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 25
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=10)
    def test_opted_out_user_was_pulled_in_by_rollout(self, mock_uniform: Mock) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 25
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        # User3 is opted out, but is pulled into both experiments by the 10% rollout
        result = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=10)
    def test_opted_out_user_was_pulled_in_by_rollout_excl_nondefault(
        self, mock_uniform: Mock
    ) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 25
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        # User3 is opted out, but is pulled into default experiments by the 10% rollout
        result = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=10)
    def test_opted_out_user_was_pulled_in_by_rollout_filter_exp(
        self, mock_uniform: Mock
    ) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 25
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        # User3 is opted out, but is pulled into default experiments by the 10% rollout
        result = rd.get_runner_prefix(
            settings_text, ["User3"], USER_BRANCH, frozenset(["otherExp"])
        )
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=25)
    def test_opted_out_user_was_pulled_out_by_rollout_filter_exp(
        self, mock_uniform: Mock
    ) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 10
            otherExp:
                rollout_perc: 50
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        # User3 is opted out, but is pulled into default experiments by the 10% rollout
        result = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    def test_lf_prefix_always_comes_first(self) -> None:
        settings_text = """
        experiments:
            otherExp:
                rollout_perc: 0
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf
        @User2,otherExp,lf

        """

        result = rd.get_runner_prefix(settings_text, ["User2"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    def test_ignores_commented_users(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        #@User1,lf
        @User2,lf,otherExp

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    def test_ignores_extra_experiments(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
            foo:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,otherExp,foo

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    def test_disables_experiment_on_exception_branches_when_not_explicitly_opted_in(
        self,
    ) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
        ---

        Users:
        @User,lf,otherExp

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], EXCEPTION_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    def test_allows_experiment_on_exception_branches_when_explicitly_opted_in(
        self,
    ) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
                all_branches: true
        ---

        Users:
        @User,lf,otherExp

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], EXCEPTION_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=5)
    def test_opted_in_user_with_rollout_perc_enabled(self, mock_uniform: Mock) -> None:
        """User opted in with 10% rollout, random=5 -> enabled"""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf:10

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=50)
    def test_opted_in_user_with_rollout_perc_disabled(self, mock_uniform: Mock) -> None:
        """User opted in with 10% rollout, random=50 -> disabled"""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf:10

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    def test_opted_in_user_without_rollout_perc_always_enabled(self) -> None:
        """User opted in without percentage (default 100%) -> always enabled"""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=15)
    def test_multiple_requesters_uses_min_perc(self, mock_uniform: Mock) -> None:
        """Two requesters with different rollout_percs, uses the minimum (10%)."""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf:10
        @User2,lf:50

        """

        # random=15, min_perc=10 -> 15 > 10 -> disabled
        result = rd.get_runner_prefix(settings_text, ["User1", "User2"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=5)
    def test_multiple_requesters_uses_min_perc_enabled(
        self, mock_uniform: Mock
    ) -> None:
        """Two requesters with different rollout_percs, min=10%, random=5 -> enabled."""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf:10
        @User2,lf:50

        """

        result = rd.get_runner_prefix(settings_text, ["User1", "User2"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    def test_opt_out_overrides_rollout_perc(self) -> None:
        """Opt-out (-lf) wins over opt-in with rollout_perc (lf:50)."""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
        ---

        Users:
        @User1,-lf,lf:50

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=5)
    def test_opted_in_user_with_rollout_perc_two_experiments(
        self, mock_uniform: Mock
    ) -> None:
        """User opted into lf at 100% and otherExp at 10%, random=5 -> both enabled"""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,otherExp:10

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=50)
    def test_opted_in_user_with_rollout_perc_partial_enable(
        self, mock_uniform: Mock
    ) -> None:
        """User opted into lf at 100% and otherExp at 10%, random=50 -> only lf enabled"""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,otherExp:10

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    def test_opted_in_user_with_zero_rollout_perc(self) -> None:
        """User opted in with 0% rollout -> never enabled"""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf:0

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")


class TestRunnerDeterminatorAmdSandboxExperiment(TestCase):
    AMD_SANDBOX_SETTINGS = """
        experiments:
            amd-sandbox:
                rollout_perc: 0
        ---

        Users:
        @User1,amd-sandbox
        @User2,lf

        """

    def test_amd_sandbox_opted_in_returns_prefix(self) -> None:
        result = rd.get_runner_prefix(self.AMD_SANDBOX_SETTINGS, ["User1"], USER_BRANCH)
        self.assertEqual("amd-sandbox-", result.amd_sandbox_prefix)
        # amd-sandbox is exposed via its own output; the base prefix is the default fleet
        self.assertEqual("mt-", result.prefix)

    def test_amd_sandbox_not_enabled_returns_default_fleet(self) -> None:
        # User2 opts into lf, but lf is not defined here, so it falls back to Meta
        result = rd.get_runner_prefix(self.AMD_SANDBOX_SETTINGS, ["User2"], USER_BRANCH)
        self.assertEqual("", result.amd_sandbox_prefix)
        self.assertEqual("mt-", result.prefix)

    def test_amd_sandbox_with_lf_keeps_both(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            amd-sandbox:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,amd-sandbox

        """
        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix)
        self.assertEqual("amd-sandbox-", result.amd_sandbox_prefix)


class TestRunnerDeterminatorAmdDpxExperiment(TestCase):
    AMD_DPX_SETTINGS = """
        experiments:
            amd-dpx:
                rollout_perc: 0
        ---

        Users:
        @User1,amd-dpx
        @User2,lf

        """

    def test_amd_dpx_opted_in_returns_prefix(self) -> None:
        result = rd.get_runner_prefix(self.AMD_DPX_SETTINGS, ["User1"], USER_BRANCH)
        self.assertEqual("amd-dpx-", result.amd_dpx_prefix)
        self.assertEqual("mt-", result.prefix)

    def test_amd_dpx_not_enabled_returns_default_fleet(self) -> None:
        result = rd.get_runner_prefix(self.AMD_DPX_SETTINGS, ["User2"], USER_BRANCH)
        self.assertEqual("", result.amd_dpx_prefix)
        self.assertEqual("mt-", result.prefix)

    def test_amd_dpx_with_lf_keeps_both(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            amd-dpx:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,amd-dpx

        """
        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix)
        self.assertEqual("amd-dpx-", result.amd_dpx_prefix)


class TestRunnerDeterminatorNoRunnerExperimentsLabel(TestCase):
    """no-runner-experiments opts out of lf, so the run stays on the default Meta fleet."""

    LF_ONLY = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf

        """

    def test_opt_out_lf_returns_meta(self) -> None:
        result = rd.get_runner_prefix(
            self.LF_ONLY,
            ["User1"],
            USER_BRANCH,
            opt_out_experiments=frozenset({"lf"}),
        )
        self.assertEqual("mt-", result.prefix)

    def test_without_opt_out_returns_lf(self) -> None:
        result = rd.get_runner_prefix(self.LF_ONLY, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix)

    def _run_main(self, *, labels: list[str], settings: str) -> dict[str, str]:
        args = Namespace(
            github_token="t",
            github_issue_repo="pytorch/test-infra",
            github_repo="pytorch/pytorch",
            github_issue=5132,
            github_actor="User1",
            github_issue_owner="User1",
            github_branch=USER_BRANCH,
            github_ref_type="branch",
            eligible_experiments=frozenset({"lf"}),
            opt_out_experiments=frozenset(),
            pr_number="123",
            workflow_name="pull",
            # No lf_allowlist assertions in this class; /dev/null reads as
            # empty YAML, which resolves to the unrestricted default.
            arc_yaml_path="/dev/null",
        )
        captured: dict[str, str] = {}
        with (
            patch.object(rd, "parse_args", return_value=args),
            patch.object(rd, "get_labels", return_value=set(labels)),
            patch.object(rd, "get_rollout_state_from_issue", return_value=settings),
            patch.object(rd, "get_potential_pr_author", return_value="User1"),
            patch.object(rd, "set_github_output", side_effect=captured.__setitem__),
        ):
            rd.main()
        return captured

    def test_main_label_disables_lf_uses_meta(self) -> None:
        out = self._run_main(labels=[rd.OPT_OUT_LABEL], settings=self.LF_ONLY)
        self.assertEqual("mt-", out[rd.GH_OUTPUT_KEY_LABEL_TYPE])

    def test_main_no_label_keeps_lf(self) -> None:
        out = self._run_main(labels=[], settings=self.LF_ONLY)
        self.assertEqual("lf-", out[rd.GH_OUTPUT_KEY_LABEL_TYPE])


class TestGetLfRunnersOutput(TestCase):
    """get_lf_runners_output: the --lf-runners allowlist (ci-infra#1081)."""

    @staticmethod
    def _arc_yaml(tmp_dir: str, body: str) -> str:
        path = Path(tmp_dir) / "arc.yaml"
        path.write_text(body)
        return str(path)

    def test_lf_disabled_returns_empty(self) -> None:
        # arc.yaml is never opened when lf is disabled.
        self.assertEqual("", rd.get_lf_runners_output("/nonexistent", lf_enabled=False))

    def test_restrict_runners_false_is_kill_switch(self) -> None:
        # The kill-switch short-circuits before arc.yaml is opened.
        self.assertEqual(
            "",
            rd.get_lf_runners_output(
                "/nonexistent", lf_enabled=True, restrict_runners=False
            ),
        )

    def test_mode_all_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            arc_yaml = self._arc_yaml(
                d, "lf_allowlist:\n  mode: all\n  runners: [l-x86iavx512-8-64]\n"
            )
            self.assertEqual("", rd.get_lf_runners_output(arc_yaml, lf_enabled=True))

    def test_mode_restricted_returns_sorted_comma_joined_runners(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            arc_yaml = self._arc_yaml(
                d,
                "lf_allowlist:\n"
                "  mode: restricted\n"
                "  runners: [l-x86iavx512-16-128, l-x86iavx512-8-64]\n",
            )
            self.assertEqual(
                "l-x86iavx512-16-128,l-x86iavx512-8-64",
                rd.get_lf_runners_output(arc_yaml, lf_enabled=True),
            )

    def test_restrict_runners_false_overrides_restricted_arc_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            arc_yaml = self._arc_yaml(
                d, "lf_allowlist:\n  mode: restricted\n  runners: [l-x86iavx512-8-64]\n"
            )
            self.assertEqual(
                "",
                rd.get_lf_runners_output(
                    arc_yaml, lf_enabled=True, restrict_runners=False
                ),
            )

    def test_missing_lf_allowlist_section_defaults_to_all(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            arc_yaml = self._arc_yaml(d, "runner_mapping: {}\n")
            self.assertEqual("", rd.get_lf_runners_output(arc_yaml, lf_enabled=True))

    def test_missing_arc_yaml_file_returns_empty(self) -> None:
        self.assertEqual(
            "", rd.get_lf_runners_output("/nonexistent/arc.yaml", lf_enabled=True)
        )

    def test_invalid_mode_returns_empty_and_logs_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            arc_yaml = self._arc_yaml(d, "lf_allowlist:\n  mode: bogus\n")
            with self.assertLogs(rd.log, level="ERROR") as logs:
                result = rd.get_lf_runners_output(arc_yaml, lf_enabled=True)
            self.assertEqual("", result)
            self.assertIn("must be one of", logs.output[0])


class TestRunnerDeterminatorLfRestrictRunners(TestCase):
    """restrict_runners kill-switch propagation (ci-infra#1081, test-infra#5132)."""

    def test_default_restrict_runners_is_true(self) -> None:
        settings_text = """
            experiments:
                lf:
                    rollout_perc: 100
            ---

            Users:

            """
        result = rd.get_runner_prefix(settings_text, [], USER_BRANCH)
        self.assertEqual("lf-", result.prefix)
        self.assertTrue(result.lf_restrict_runners)

    def test_restrict_runners_false_is_kill_switch(self) -> None:
        settings_text = """
            experiments:
                lf:
                    rollout_perc: 100
                    restrict_runners: false
            ---

            Users:

            """
        result = rd.get_runner_prefix(settings_text, [], USER_BRANCH)
        self.assertEqual("lf-", result.prefix)
        self.assertFalse(result.lf_restrict_runners)

    def test_lf_not_enabled_keeps_default_true(self) -> None:
        # restrict_runners is only read from the experiment when lf itself
        # enables; an unused kill-switch setting must not leak through.
        settings_text = """
            experiments:
                lf:
                    rollout_perc: 0
                    restrict_runners: false
            ---

            Users:

            """
        result = rd.get_runner_prefix(settings_text, [], USER_BRANCH)
        self.assertEqual("mt-", result.prefix)
        self.assertTrue(result.lf_restrict_runners)


class TestRunnerDeterminatorMainLfRunnersOutput(TestCase):
    """main()'s end-to-end plumbing: get_runner_prefix -> get_lf_runners_output
    -> the lf-runners GITHUB_OUTPUT (ci-infra#1081)."""

    RESTRICTED_ARC_YAML = (
        "lf_allowlist:\n  mode: restricted\n  runners: [l-x86iavx512-8-64]\n"
    )

    def _run_main(self, *, settings: str, arc_yaml_path: str) -> dict[str, str]:
        args = Namespace(
            github_token="t",
            github_issue_repo="pytorch/test-infra",
            github_repo="pytorch/pytorch",
            github_issue=5132,
            github_actor="User1",
            github_issue_owner="User1",
            github_branch=USER_BRANCH,
            github_ref_type="branch",
            eligible_experiments=frozenset({"lf"}),
            opt_out_experiments=frozenset(),
            pr_number="",
            workflow_name="pull",
            arc_yaml_path=arc_yaml_path,
        )
        captured: dict[str, str] = {}
        with (
            patch.object(rd, "parse_args", return_value=args),
            patch.object(rd, "get_rollout_state_from_issue", return_value=settings),
            patch.object(rd, "get_potential_pr_author", return_value="User1"),
            patch.object(rd, "set_github_output", side_effect=captured.__setitem__),
        ):
            rd.main()
        return captured

    def test_restricted_arc_yaml_flows_through_to_output(self) -> None:
        settings_text = """
            experiments:
                lf:
                    rollout_perc: 100
            ---

            Users:

            """
        with tempfile.TemporaryDirectory() as d:
            arc_yaml = Path(d) / "arc.yaml"
            arc_yaml.write_text(self.RESTRICTED_ARC_YAML)
            out = self._run_main(settings=settings_text, arc_yaml_path=str(arc_yaml))
        self.assertEqual("lf-", out[rd.GH_OUTPUT_KEY_LABEL_TYPE])
        self.assertEqual("l-x86iavx512-8-64", out[rd.GH_OUTPUT_KEY_LF_RUNNERS])

    def test_kill_switch_overrides_restricted_arc_yaml(self) -> None:
        settings_text = """
            experiments:
                lf:
                    rollout_perc: 100
                    restrict_runners: false
            ---

            Users:

            """
        with tempfile.TemporaryDirectory() as d:
            arc_yaml = Path(d) / "arc.yaml"
            arc_yaml.write_text(self.RESTRICTED_ARC_YAML)
            out = self._run_main(settings=settings_text, arc_yaml_path=str(arc_yaml))
        self.assertEqual("lf-", out[rd.GH_OUTPUT_KEY_LABEL_TYPE])
        self.assertEqual("", out[rd.GH_OUTPUT_KEY_LF_RUNNERS])

    def test_meta_fleet_never_gets_lf_runners(self) -> None:
        settings_text = """
            experiments:
                lf:
                    rollout_perc: 0
            ---

            Users:

            """
        with tempfile.TemporaryDirectory() as d:
            arc_yaml = Path(d) / "arc.yaml"
            arc_yaml.write_text(self.RESTRICTED_ARC_YAML)
            out = self._run_main(settings=settings_text, arc_yaml_path=str(arc_yaml))
        self.assertEqual("mt-", out[rd.GH_OUTPUT_KEY_LABEL_TYPE])
        self.assertEqual("", out[rd.GH_OUTPUT_KEY_LF_RUNNERS])


if __name__ == "__main__":
    main()


class TestScaleConfigPrefix(TestCase):
    """Only wincanary/wincanarylf drive scale-config-label-type."""

    SETTINGS = """
experiments:
  lf:
    rollout_perc: 0
  wincanary:
    rollout_perc: 0
  wincanarylf:
    rollout_perc: 0
---
@lfuser,lf
@wcuser,wincanary
@wclfuser,wincanarylf
@bothuser,lf,wincanarylf
@plainuser,
"""
    ALL = frozenset({"lf", "wincanary", "wincanarylf"})

    def _result(self, user: str) -> rd.RunnerPrefixResult:
        return rd.get_runner_prefix(
            self.SETTINGS, (user, user), "somebranch", self.ALL, frozenset()
        )

    def test_no_variant_means_no_prefix(self) -> None:
        self.assertEqual("", self._result("plainuser").scale_config_prefix)

    def test_lf_alone_does_not_set_a_prefix(self) -> None:
        # lf rolls out over ALL workflows; it must not relocate Windows builds.
        self.assertEqual("", self._result("lfuser").scale_config_prefix)

    def test_wincanary(self) -> None:
        self.assertEqual("wincanary.", self._result("wcuser").scale_config_prefix)

    def test_wincanarylf(self) -> None:
        self.assertEqual("wincanarylf.", self._result("wclfuser").scale_config_prefix)

    def test_lf_does_not_compose_with_the_variant(self) -> None:
        # never "lf.wincanarylf."
        self.assertEqual("wincanarylf.", self._result("bothuser").scale_config_prefix)

    def test_arc_label_type_is_untouched(self) -> None:
        self.assertEqual("mt-", self._result("plainuser").prefix)
        self.assertEqual("lf-", self._result("lfuser").prefix)
        self.assertEqual("lf-", self._result("bothuser").prefix)
