# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import context, test_util
from threading import Thread


@context.define_context()
class MyContext(object):
    pass


class TestContext(test_util.TestCase):
    def use_my_context(self):
        try:
            for _ in range(100):
                with MyContext() as a:
                    for _ in range(100):
                        self.assertTrue(MyContext.current() == a)
        except Exception as e:
            self._exceptions.append(e)

    def testMultiThreaded(self):
        threads = []
        self._exceptions = []
        for _ in range(8):
            thread = Thread(target=self.use_my_context)
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()
        for e in self._exceptions:
            raise e

    @MyContext()
    def testDecorator(self):
        self.assertIsNotNone(MyContext.current())
