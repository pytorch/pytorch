Expiration Timers
==================

.. automodule:: torch.distributed.elastic.timer
.. currentmodule:: torch.distributed.elastic.timer

Client Methods
---------------
.. autofunction:: torch.distributed.elastic.timer.configure

.. autofunction:: torch.distributed.elastic.timer.expires

Server/Client Implementations
------------------------------
Below are the timer server and client pairs that are provided by torchelastic.

.. note:: Timer server and clients always have to be implemented and used
          in pairs since there is a messaging protocol between the server
          and client.

Below is a pair of timer server and client that is implemented based on
a ``multiprocess.Queue``.

.. autoclass:: LocalTimerServer

.. autoclass:: LocalTimerClient

Below is another pair of timer server and client that is implemented
based on a named pipe.

.. autoclass:: FileTimerServer

.. autoclass:: FileTimerClient


Writing a custom timer server/client
--------------------------------------

To write your own timer server and client extend the
``torch.distributed.elastic.timer.TimerServer`` for the server and
``torch.distributed.elastic.timer.TimerClient`` for the client. The
``TimerRequest`` object is used to pass messages between
the server and client.

.. autoclass:: TimerRequest
   :members:

.. autoclass:: TimerServer
   :members:

.. autoclass:: TimerClient
   :members:


Debug info logging
-------------------

.. automodule:: torch.distributed.elastic.timer.debug_info_logging

.. autofunction:: torch.distributed.elastic.timer.debug_info_logging.log_debug_info_for_expired_timers
