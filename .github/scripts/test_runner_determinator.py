from argparse import Namespace
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


class TestRunnerDeterminatorAmdDoExperiment(TestCase):
    AMD_DO_SETTINGS = """
        experiments:
            amd-do:
                rollout_perc: 0
        ---

        Users:
        @User1,amd-do
        @User2,lf

        """

    def test_amd_do_opted_in_returns_prefix(self) -> None:
        result = rd.get_runner_prefix(self.AMD_DO_SETTINGS, ["User1"], USER_BRANCH)
        self.assertEqual("amd-do-", result.amd_do_prefix)
        # amd-do is exposed via its own output; the base prefix is the default fleet
        self.assertEqual("mt-", result.prefix)

    def test_amd_do_not_enabled_returns_default_fleet(self) -> None:
        # User2 opts into lf, but lf is not defined here, so it falls back to Meta
        result = rd.get_runner_prefix(self.AMD_DO_SETTINGS, ["User2"], USER_BRANCH)
        self.assertEqual("", result.amd_do_prefix)
        self.assertEqual("mt-", result.prefix)

    def test_amd_do_with_lf_keeps_both(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            amd-do:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,amd-do

        """
        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix)
        self.assertEqual("amd-do-", result.amd_do_prefix)


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


if __name__ == "__main__":
    main()
