




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
