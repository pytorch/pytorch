import multiprocessing

# Global properties of our device
NUM_DEVICES = 7

class _Daemon():
    def __init__(self):
        super().__init__()
        self.req_queue = multiprocessing.Queue()
        self.ans_queue = multiprocessing.Queue()

        self.runner = multiprocessing.Process(target=self.run_forever, args=(self.req_queue, self.ans_queue), daemon=True)
        self.runner.start()

    def exec(self, cmd, *args):
        self.req_queue.put((cmd,) + args)
        res = self.ans_queue.get()
        if res == "ERROR":
            raise RuntimeError("Error in daemon, see print")
        else:
            return res

    @staticmethod
    def run_forever(req_queue, ans_queue):
        while True:
            cmd, *args = req_queue.get()
            print("Command: ", cmd)
            if cmd == "deviceCount":
                assert len(args) == 0
                ans_queue.put(NUM_DEVICES)
            else:
                print("Bad command in worker")
                ans_queue.put("ERROR")

daemon = _Daemon()

