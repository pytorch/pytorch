# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Expiration timers are set up on the same process as the agent and
used from your script to deal with stuck workers. When you go into
a code-block that has the potential to get stuck you can acquire
an expiration timer, which instructs the timer server to kill the
process if it does not release the timer by the self-imposed expiration
deadline.

Usage::

    import torchelastic.timer as timer
    import torchelastic.agent.server as agent

    def main():
        start_method = "spawn"
        message_queue = mp.get_context(start_method).Queue()
        server = timer.LocalTimerServer(message, max_interval=0.01)
        server.start() # non-blocking

        spec = WorkerSpec(
                    fn=trainer_func,
                    args=(message_queue,),
                    ...<OTHER_PARAMS...>)
        agent = agent.LocalElasticAgent(spec, start_method)
        agent.run()

    def trainer_func(message_queue):
        timer.configure(timer.LocalTimerClient(message_queue))
        with timer.expires(after=60): # 60 second expiry
            # do some work

In the example above if ``trainer_func`` takes more than 60 seconds to
complete, then the worker process is killed and the agent retries the worker group.
"""

from .api import TimerClient, TimerRequest, TimerServer, configure, expires  # noqa: F401
from .local_timer import LocalTimerClient, LocalTimerServer  # noqa: F401
from .file_based_local_timer import FileTimerClient, FileTimerServer, FileTimerRequest  # noqa: F401
