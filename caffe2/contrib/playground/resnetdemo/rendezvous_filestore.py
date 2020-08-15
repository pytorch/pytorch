from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from caffe2.python import dyndep
dyndep.InitOpsLibrary('@/caffe2/caffe2/distributed:file_store_handler_ops')


# rendezvous should NOT be unique for each operator.  It should have
# the same run_id on different operators.  say we have two shards,
# both shards created rendezvous of run_id "aaa_bbb_epoch_09", and this
# rendezvous will wait for two shards to join because max_shards is specified
# to be 2.  If each shard created an rendezvous with different run_id,
# each of them are waiting for different rendezvous to join, they will
# never wait for each other and therefore timeout eventually.

def gen_rendezvous_ctx(self, model, dataset, is_train):
    if self.opts['distributed']['num_shards'] < 2:
        return None
    # have issue when try to set this up on more shards
    workspace.RunOperatorOnce(
        core.CreateOperator(
            "FileStoreHandlerCreate", [], ["store_handler"],
            path="/tmp",
            prefix="epoch.{}".format(self.epoch),
        )
    )

    rendezvous = dict(
        kv_handler="store_handler",
        shard_id=self.shard_id,
        num_shards=self.opts['distributed']['num_shards'],
        engine="GLOO",
        # transport=args.distributed_transport,
        transport="tcp",
        # interface=interfaces[0],
        interface=[],
        exit_nets=None) if is_train else None
    return rendezvous
