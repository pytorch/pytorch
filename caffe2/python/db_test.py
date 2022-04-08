




from caffe2.python import workspace

import os
import tempfile
import unittest


class TestDB(unittest.TestCase):
    def setUp(self) -> None:
        handle, self.file_name = tempfile.mkstemp()
        os.close(handle)
        self.data = [
            (
                "key{}".format(i).encode("ascii"),
                "value{}".format(i).encode("ascii")
            )
            for i in range(1, 10)
        ]

    def testSimple(self) -> None:
        # pyre-fixme[16]: Module `_import_c_extension` has no attribute `create_db`.
        db = workspace.C.create_db(
            # pyre-fixme[16]: Module `_import_c_extension` has no attribute `Mode`.
            "minidb", self.file_name, workspace.C.Mode.write)

        for key, value in self.data:
            transaction = db.new_transaction()
            transaction.put(key, value)
            del transaction

        del db  # should close DB

        # pyre-fixme[16]: Module `_import_c_extension` has no attribute `create_db`.
        db = workspace.C.create_db(
            # pyre-fixme[16]: Module `_import_c_extension` has no attribute `Mode`.
            "minidb", self.file_name, workspace.C.Mode.read)
        cursor = db.new_cursor()
        data = []
        while cursor.valid():
            data.append((cursor.key(), cursor.value()))
            cursor.next()  # noqa: B305
        del cursor

        db.close()  # test explicit db closer
        self.assertEqual(data, self.data)
