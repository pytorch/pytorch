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
        self.assertDictEqual(
            {"User1": ["lf"], "User2": ["lf", "otherExp"]},
            users,
            "Users not parsed correctly",
        )

    def test_parse_users_without_settings(self) -> None:
        settings_text = """

        @User1,lf
        @User2,lf,otherExp

        """

        users = rd.parse_users(settings_text)
        self.assertDictEqual(
            {"User1": ["lf"], "User2": ["lf", "otherExp"]},
            users,
            "Users not parsed correctly",
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
        prefix = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf.", prefix, "Runner prefix not correct for User1")

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
        prefix = rd.get_runner_prefix(settings_text, ["User2"], USER_BRANCH)
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for User2")

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
        prefix = rd.get_runner_prefix(settings_text, ["User2"], USER_BRANCH)
        self.assertEqual("lf.", prefix, "Runner prefix not correct for User2")

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
        prefix = rd.get_runner_prefix(
            settings_text, ["User2"], USER_BRANCH, frozenset(["lf", "otherExp"])
        )
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for User2")

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
        prefix = rd.get_runner_prefix(
            settings_text, ["User2"], USER_BRANCH, frozenset(["otherExp"])
        )
        self.assertEqual("otherExp.", prefix, "Runner prefix not correct for User2")

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
        prefix = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("", prefix, "Runner prefix not correct for user")

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
        prefix = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for user")

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
        prefix = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("lf.", prefix, "Runner prefix not correct for user")

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
        prefix = rd.get_runner_prefix(
            settings_text, ["User3"], USER_BRANCH, frozenset(["otherExp"])
        )
        self.assertEqual("otherExp.", prefix, "Runner prefix not correct for user")

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
        prefix = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("", prefix, "Runner prefix not correct for user")

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

        prefix = rd.get_runner_prefix(settings_text, ["User2"], USER_BRANCH)
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for user")

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

        prefix = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("", prefix, "Runner prefix not correct for user")

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

        prefix = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for user")

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

        prefix = rd.get_runner_prefix(settings_text, ["User1"], EXCEPTION_BRANCH)
        self.assertEqual("", prefix, "Runner prefix not correct for user")

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

        prefix = rd.get_runner_prefix(settings_text, ["User1"], EXCEPTION_BRANCH)
        self.assertEqual("lf.", prefix, "Runner prefix not correct for user")


if __name__ == "__main__":
    main()
