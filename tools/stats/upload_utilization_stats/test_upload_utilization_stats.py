import unittest
from unittest.mock import patch
from datetime import datetime
from collections import Counter
import os
import sys

# python script is mainly for uploading test logs to s3 for a test job
# adding sys.path makes the monitor script able to import path tools.stats.utilization_stats_lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from tools.stats.utilization_stats_lib import UtilizationRecord
from tools.stats.upload_utilization_stats.upload_utilization_stats import (
    SegmentGenerator,
)

# timestamp January 1, 2022 12:00:00
TEST_TIME_STAMP_1 = datetime(2022, 1, 1, 12, 0, 0).timestamp()
PYTEST1_NAME = "python test1.py"
PYPIP_INSTALL_NAME = "python pip install install1"

class TestSegmentGenerator(unittest.TestCase):
    def test_generate_empty_records(self):
        records = []

        # execute
        generator = SegmentGenerator()
        segments = generator.generate(records)

        # assert
        self.assertEqual(segments, [])

    def test_generate_single_record(self):
        record = UtilizationRecord(
            timestamp=TEST_TIME_STAMP_1, cmd_names=[PYTEST1_NAME], level="PYTHON_CMD"
        )
        records = [record]

        # execute
        generator = SegmentGenerator()
        segments = generator.generate(records)

        # assert
        self.assertEqual(len(segments), 1)

    def test_generate_single_record_with_multiple_cmds(self):
        record = UtilizationRecord(
            timestamp=TEST_TIME_STAMP_1,
            cmd_names=[PYTEST1_NAME, PYPIP_INSTALL_NAME],
            level="PYTHON_CMD",
        )
        records = [record]

        # execute
        generator = SegmentGenerator()
        segments = generator.generate(records)

        # assert
        self.assertEqual(len(segments), 2)

    def test_generate_multiple_records_overlaping(self):
        record1 = UtilizationRecord(
            timestamp=TEST_TIME_STAMP_1, cmd_names=[PYTEST1_NAME], level="PYTHON_CMD"
        )
        record2 = UtilizationRecord(
            timestamp=datetime(2022, 1, 1, 12, 0, 30).timestamp(),
            cmd_names=[PYTEST1_NAME],
            level="PYTHON_CMD",
        )
        record3 = UtilizationRecord(
            timestamp=datetime(2022, 1, 1, 12, 1, 15).timestamp(),
            cmd_names=[PYTEST1_NAME, PYPIP_INSTALL_NAME],
            level="PYTHON_CMD",
        )
        record4 = UtilizationRecord(
            timestamp=datetime(2022, 1, 1, 12, 1, 31).timestamp(),
            cmd_names=[PYTEST1_NAME, PYPIP_INSTALL_NAME],
            level="PYTHON_CMD",
        )
        record5 = UtilizationRecord(
            timestamp=datetime(2022, 1, 1, 12, 1, 42).timestamp(),
            cmd_names=[PYTEST1_NAME, PYPIP_INSTALL_NAME],
            level="PYTHON_CMD",
        )
        records = [record1, record2, record3, record4, record5]

        # execute
        generator = SegmentGenerator()
        segments = generator.generate(records)

        # assert
        self.assertEqual(len(segments), 2)

    def test_generate_multiple_segments_larger_than_default_threshold_setting(self):
        record1 = UtilizationRecord(
            timestamp=TEST_TIME_STAMP_1,
            cmd_names=[PYTEST1_NAME, PYPIP_INSTALL_NAME],
            level="PYTHON_CMD",
        )
        record2 = UtilizationRecord(
            timestamp=datetime(2022, 1, 1, 12, 0, 30).timestamp(),
            cmd_names=[PYTEST1_NAME],
            level="PYTHON_CMD",
        )
        record3 = UtilizationRecord(
            timestamp=datetime(2022, 1, 1, 12, 1, 15).timestamp(),
            cmd_names=[PYTEST1_NAME],
            level="PYTHON_CMD",
        )
        # record has more than 1 minute gap
        record4 = UtilizationRecord(
            timestamp=datetime(2022, 1, 1, 12, 5, 15).timestamp(),
            cmd_names=[PYTEST1_NAME],
            level="PYTHON_CMD",
        )
        record5 = UtilizationRecord(
            timestamp=datetime(2022, 1, 1, 12, 5, 18).timestamp(),
            cmd_names=[PYTEST1_NAME],
            level="PYTHON_CMD",
        )
        records = [record1, record2, record3, record4, record5]

        # execute
        generator = SegmentGenerator()
        segments = generator.generate(records)

        # assert
        counter = Counter(seg.name for seg in segments)
        self.assertEqual(counter[PYTEST1_NAME], 2)
        self.assertEqual(counter[PYPIP_INSTALL_NAME], 1)
        self.assertEqual(len(segments), 3)

    def test_generate_multiple_segments_with_customized_threshold(self):
        # set threshold to consider as continuous segment to 10 seconds
        test_threshold = 10

        record1 = UtilizationRecord(
            timestamp=TEST_TIME_STAMP_1,
            cmd_names=[PYTEST1_NAME, PYPIP_INSTALL_NAME],
            level="PYTHON_CMD",
        )
        record2 = UtilizationRecord(
            timestamp=datetime(2022, 1, 1, 12, 0, 20).timestamp(),
            cmd_names=[PYTEST1_NAME],
            level="PYTHON_CMD",
        )
        record3 = UtilizationRecord(
            timestamp=datetime(2022, 1, 1, 12, 1, 15).timestamp(),
            cmd_names=[PYTEST1_NAME],
            level="PYTHON_CMD",
        )
        # record has more than 1 minute gap
        record4 = UtilizationRecord(
            timestamp=datetime(2022, 1, 1, 12, 5, 15).timestamp(),
            cmd_names=[PYTEST1_NAME],
            level="PYTHON_CMD",
        )
        record5 = UtilizationRecord(
            timestamp=datetime(2022, 1, 1, 12, 5, 18).timestamp(),
            cmd_names=[PYTEST1_NAME],
            level="PYTHON_CMD",
        )
        records = [record1, record2, record3, record4, record5]

        # execute
        generator = SegmentGenerator()
        segments = generator.generate(records, test_threshold)

        # assert
        counter = Counter(seg.name for seg in segments)
        self.assertEqual(counter[PYTEST1_NAME], 4)
        self.assertEqual(counter[PYPIP_INSTALL_NAME], 1)
        self.assertEqual(len(segments), 5)


if __name__ == "__main__":
    unittest.main()
