from unittest import main, TestCase
from unittest.mock import patch
import runner_determinator as rd


class TestRunnerDeterminatorIssueParser(TestCase):
    def test_parse_settings(self):
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 0
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
            rd.Experiment(rollout_perc=0),
            settings.experiments["otherExp"],
            "otherExp settings not parsed correctly",
        )

    def test_parse_users(self):
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
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

    def test_parse_users_without_settings(self):
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

    def test_opted_in_user(self):
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
        prefix = rd.get_runner_prefix(settings_text, ["User1"])
        self.assertEqual("lf.", prefix, "Runner prefix not correct for User1")

    def test_opted_in_user_two_experiments(self):
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
        prefix = rd.get_runner_prefix(settings_text, ["User2"])
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for User1")

    @patch("random.randint", return_value=50)
    def test_opted_out_user(self, mock_randint):
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
        prefix = rd.get_runner_prefix(settings_text, ["User3"])
        self.assertEqual("", prefix, "Runner prefix not correct for user")

    @patch("random.randint", return_value=10)
    def test_opted_out_user_was_pulled_in_by_rollout(
        self, mock_randint
    ):
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
        prefix = rd.get_runner_prefix(settings_text, ["User3"])
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for user")

    def test_lf_prefix_always_comes_first(
        self
    ):
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

        prefix = rd.get_runner_prefix(settings_text, ["User2"])
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for user")

    def test_ignores_commented_users(
        self
    ):
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

        prefix = rd.get_runner_prefix(settings_text, ["User1"])
        self.assertEqual("", prefix, "Runner prefix not correct for user")

    def test_ignores_extra_experiments(
        self
    ):
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

        prefix = rd.get_runner_prefix(settings_text, ["User1"])
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for user")



if __name__ == "__main__":
    main()
