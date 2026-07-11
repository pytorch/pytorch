# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import enum


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PFull = enum.auto()
    PEmpty = enum.auto()


class NamedBarrierFwdSm100(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    TmemPtr = enum.auto()
    SoftmaxStatsW0 = enum.auto()
    SoftmaxStatsW1 = enum.auto()
    SoftmaxStatsW2 = enum.auto()
    SoftmaxStatsW3 = enum.auto()
    SoftmaxStatsW4 = enum.auto()
    SoftmaxStatsW5 = enum.auto()
    SoftmaxStatsW6 = enum.auto()
    SoftmaxStatsW7 = enum.auto()


class NamedBarrierBwd(enum.IntEnum):
    Epilogue = enum.auto()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PdS = enum.auto()
    dQFullWG0 = enum.auto()
    dQFullWG1 = enum.auto()
    dQFullWG2 = enum.auto()
    dQEmptyWG0 = enum.auto()
    dQEmptyWG1 = enum.auto()
    dQEmptyWG2 = enum.auto()


class NamedBarrierBwdSm100(enum.IntEnum):
    EpilogueWG1 = enum.auto()
    EpilogueWG2 = enum.auto()
    Compute = enum.auto()
    dQaccReduce = enum.auto()
    TmemPtr = enum.auto()


class NamedBarrierFwdSm100_MLA2CTA(enum.IntEnum):
    Epilogue = enum.auto()
    TmemPtr = enum.auto()
    Cpasync = enum.auto()
    Softmax = enum.auto()
    SoftmaxStatsFull = enum.auto()
    SoftmaxStatsEmpty = enum.auto()


class NamedBarrierBwdSm100_MLA2CTA(enum.IntEnum):
    Epilogue = enum.auto()
    TmemPtr = enum.auto()
    Cpasync = enum.auto()
    Softmax = enum.auto()
