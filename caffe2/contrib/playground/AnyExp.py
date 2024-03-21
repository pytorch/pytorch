




from abc import abstractmethod

from caffe2.python import workspace
from caffe2.python import timeout_guard
from caffe2.python import data_parallel_model
from . import checkpoint as checkpoint

from . import ModuleRegister as ModuleRegister
from . import module_map as module_map

# instantiate logger outside of distributed operators may trigger error
# logger need to be created in each idividual operator instead.
import os
import inspect
import time
import logging
logging.basicConfig()
log = logging.getLogger("AnyExp")
log.setLevel(logging.DEBUG)


def initOpts(opts):

    workspace.GlobalInit(
        ['caffe2', '--caffe2_log_level=2', '--caffe2_gpu_memory_tracking=0'])

    assert (opts['distributed']['num_gpus'] > 0 or
            opts['distributed']['num_cpus'] > 0),\
        "Need to specify num_gpus or num_cpus to decide which device to use."

    trainWithCPU = (opts['distributed']['num_gpus'] == 0)
    num_xpus = opts['distributed']['num_cpus'] if \
        trainWithCPU else opts['distributed']['num_gpus']
    first_xpu = opts['distributed']['first_cpu_id'] if \
        trainWithCPU else opts['distributed']['first_gpu_id']
    opts['distributed']['device'] = 'cpu' if trainWithCPU else 'gpu'

    opts['model_param']['combine_spatial_bn'] =\
        trainWithCPU and opts['model_param']['combine_spatial_bn']

    opts['distributed']['num_xpus'] = num_xpus
    opts['distributed']['first_xpu_id'] = first_xpu
    opts['temp_var'] = {}
    opts['temp_var']['metrics_output'] = {}

    return opts


def initDefaultModuleMap():
    registerModuleMap(module_map)


def registerModuleMap(module_map):
    ModuleRegister.registerModuleMap(module_map)


def aquireDatasets(opts):
    myAquireDataModule = ModuleRegister.getModule(opts['input']['input_name_py'])
    return myAquireDataModule.get_input_dataset(opts)


def createTrainerClass(opts):
    return ModuleRegister.constructTrainerClass(AnyExpTrainer, opts)


def overrideAdditionalMethods(myTrainerClass, opts):
    return ModuleRegister.overrideAdditionalMethods(myTrainerClass, opts)


def initialize_params_from_file(*args, **kwargs):
    return checkpoint.initialize_params_from_file(*args, **kwargs)


class AnyExpTrainer:

    def __init__(self, opts):
        import logging
        logging.basicConfig()
        log = logging.getLogger("AnyExp")
        log.setLevel(logging.DEBUG)
        self.log = log

        self.opts = opts
        self.train_dataset = None
        self.test_dataset = None
        self.train_df = None
        self.test_df = None

        self.metrics = {}
        self.plotsIngredients = []

        self.record_epochs = []
        self.samples_per_sec = []
        self.secs_per_train = []

        self.metrics_output = opts['temp_var']['metrics_output']

        first_xpu = opts['distributed']['first_xpu_id']
        num_xpus = opts['distributed']['num_xpus']

        self.xpus = range(first_xpu, first_xpu + num_xpus)

        self.total_batch_size = \
            self.opts['epoch_iter']['batch_per_device'] * \
            self.opts['distributed']['num_xpus'] * \
            self.opts['distributed']['num_shards']
        self.epoch_iterations = \
            self.opts['epoch_iter']['num_train_sample_per_epoch'] // \
            self.total_batch_size

        if len(opts['input']['datasets']) > 0:
            self.train_df = opts['input']['datasets'][0]
            if len(opts['input']['datasets']) == 2:
                self.test_df = opts['input']['datasets'][1]
        # at this point, the intance of this class becomes many instances
        # running on different machines.  Most of their attributes are same,
        # but the shard_ids are different.
        self.shard_id = opts['temp_var']['shard_id']
        self.start_epoch = opts['temp_var']['start_epoch']
        self.epoch = opts['temp_var']['epoch']
        self.epochs_to_run = opts['epoch_iter']['num_epochs_per_flow_schedule']

        log.info('opts: {}'.format(str(opts)))

    @abstractmethod
    def get_input_dataset(self, opts):
        pass

    @abstractmethod
    def get_model_input_fun(self):
        pass

    @abstractmethod
    def init_model(self):
        pass

    def init_metrics(self):
        metrics = self.opts['output']['metrics']
        for metric in metrics:
            meterClass = self.getMeterClass(metric['meter_py'])
            # log.info('metric.meter_kargs {}'.format(metric.meter_kargs))
            # log.info('type meter_kargs {}'.format(type(metric.meter_kargs)))
            meterInstance = meterClass(opts=self.opts, **metric['meter_kargs'])
            self.add_metric(metric['name'], meterInstance, metric['is_train'])

    def getMeterClass(self, meterName):
        return ModuleRegister.getClassFromModule(meterName, meterName)

    def add_metric(self, name, calculator, is_train):
        metrics = self.metrics
        metrics[name] = {}
        metrics[name]['calculator'] = calculator
        metrics[name]['is_train'] = is_train
        metrics[name]['output'] = []

    def extendMetricsOutput(self):
        metrics_output = self.metrics_output
        if not metrics_output:
            metrics_output['epochs'] = self.record_epochs
            metrics_output['samples_per_sec'] = self.samples_per_sec
            metrics_output['secs_per_train'] = self.secs_per_train
            for metric, value in self.metrics.items():
                metrics_output[metric] = value['output']
        else:
            metrics_output['epochs'].extend(self.record_epochs)
            metrics_output['samples_per_sec'].extend(self.samples_per_sec)
            metrics_output['secs_per_train'].extend(self.secs_per_train)
            for metric, value in self.metrics.items():
                metrics_output[metric].extend(value['output'])

    @abstractmethod
    def init_plots(self):
        pass

    def add_plot(self, x, x_title, ys, y_title):
        plotsIngredients = self.plotsIngredients
        aPlotIngredients = {}
        aPlotIngredients['x'] = x
        aPlotIngredients['x_title'] = x_title
        aPlotIngredients['ys'] = ys
        aPlotIngredients['y_title'] = y_title
        plotsIngredients.append(aPlotIngredients)

    @abstractmethod
    def init_logs(self):
        pass

    def list_of_epochs(self):
        iter_end_point = min(self.opts['epoch_iter']['num_epochs'],
                             self.epoch +
                             self.opts['epoch_iter']['num_epochs_per_flow_schedule'])
        return range(self.epoch, iter_end_point)

    def list_of_epoch_iters(self):
        return range(0, self.epoch_iterations)

    @abstractmethod
    def fun_per_epoch_b4RunNet(self, epoch):
        pass

    @abstractmethod
    def fun_per_epoch_aftRunNet(self, epoch):
        pass

    def checkpoint(self, epoch):
        self.model_path = checkpoint.save_model_params(
            True, self.train_model, self.gen_checkpoint_path(True, epoch + 1),
            epoch + 1, self.opts, float('-inf'))

    def gen_checkpoint_path(self, is_checkpoint, epoch):
        if (is_checkpoint):
            filename = "model_checkpoint_epoch{}.pkl".format(epoch)
        else:
            filename = "model_final.pkl"
        return self.opts['output']['checkpoint_folder'] + filename

    # @abstractmethod
    # def gen_checkpoint_path(self, is_checkpoint, epoch):
    #     pass

    @abstractmethod
    def fun_per_iter_b4RunNet(self, epoch, epoch_iter):
        pass

    @abstractmethod
    def fun_per_iter_aftRunNetB4Test(self, epoch, epoch_iter):
        pass

    @abstractmethod
    def fun_per_iter_aftRunNetAftTest(self, epoch, epoch_iter):
        pass

    @abstractmethod
    def fun_conclude_operator(self, opts):
        pass

    def createMetricsPlotsModelsOutputs(self):
        self.extendMetricsOutput()
        self.model_output = self.model_path

    @abstractmethod
    def assembleAllOutputs(self):
        pass

    @abstractmethod
    def gen_input_builder_fun(self, model, dataset, is_train):
        pass

    @abstractmethod
    def gen_forward_pass_builder_fun(self, model, dataset, is_train):
        pass

    @abstractmethod
    def gen_param_update_builder_fun(self, model, dataset, is_train):
        pass

    @abstractmethod
    def gen_optimizer_fun(self, model, dataset, is_train):
        pass

    @abstractmethod
    def gen_rendezvous_ctx(self, model, dataset, is_train):
        pass

    @abstractmethod
    def run_training_net(self):
        pass

    @abstractmethod
    def run_testing_net(self):
        if self.test_model is None:
            return
        timeout = 2000.0
        with timeout_guard.CompleteInTimeOrDie(timeout):
            workspace.RunNet(self.test_model.net.Proto().name)

    # @abstractmethod
    def planning_output(self):
        self.init_metrics()
        self.init_plots()
        self.init_logs()

    def prep_data_parallel_models(self):
        self.prep_a_data_parallel_model(self.train_model,
                                        self.train_dataset, True)
        self.prep_a_data_parallel_model(self.test_model,
                                        self.test_dataset, False)

    def prep_a_data_parallel_model(self, model, dataset, is_train):
        if model is None:
            return

        log.info('in prep_a_data_parallel_model')

        param_update = \
            self.gen_param_update_builder_fun(model, dataset, is_train) \
            if self.gen_param_update_builder_fun is not None else None
        log.info('in prep_a_data_parallel_model param_update done ')

        optimizer = \
            self.gen_optimizer_fun(model, dataset, is_train) \
            if self.gen_optimizer_fun is not None else None
        log.info('in prep_a_data_parallel_model optimizer done ')

        max_ops = self.opts['model_param']['max_concurrent_distributed_ops']
        data_parallel_model.Parallelize(
            model,
            input_builder_fun=self.gen_input_builder_fun(model, dataset, is_train),
            forward_pass_builder_fun=self.gen_forward_pass_builder_fun(
                model, dataset, is_train),
            param_update_builder_fun=param_update,
            optimizer_builder_fun=optimizer,
            devices=self.xpus,
            rendezvous=self.gen_rendezvous_ctx(model, dataset, is_train),
            broadcast_computed_params=False,
            optimize_gradient_memory=self.opts['model_param']['memonger'],
            use_nccl=self.opts['model_param']['cuda_nccl'],
            max_concurrent_distributed_ops=max_ops,
            cpu_device=(self.opts['distributed']['device'] == 'cpu'),
            # "shared model" will only keep model parameters for cpu_0 or gpu_0
            # will cause issue when initialize each gpu_0, gpu_1, gpu_2 ...
            # shared_model=(self.opts['distributed']['device'] == 'cpu'),
            combine_spatial_bn=self.opts['model_param']['combine_spatial_bn'],
        )
        log.info('in prep_a_data_parallel_model Parallelize done ')

        # log.info("Current blobs in workspace: {}".format(workspace.Blobs()))

        workspace.RunNetOnce(model.param_init_net)
        log.info('in prep_a_data_parallel_model RunNetOnce done ')

        # for op in model.net.Proto().op:
        #     log.info('op type engine {} {}'.format(op.type, op.engine))

        log.info('model.net.Proto() {}'.format(model.net.Proto()))

        workspace.CreateNet(model.net)

        # for op in model.net.Proto().op:
        #     log.info('after CreateNet op type engine {} {}'.
        #         format(op.type, op.engine))

        log.info('in prep_a_data_parallel_model CreateNet done ')

    def loadCheckpoint(self):
        opts = self.opts
        previous_checkpoint = opts['temp_var']['checkpoint_model']
        pretrained_model = opts['temp_var']['pretrained_model']
        num_xpus = opts['distributed']['num_xpus']
        if (previous_checkpoint is not None):
            if os.path.exists(previous_checkpoint):
                log.info('Load previous checkpoint:{}'.format(
                    previous_checkpoint
                ))
                start_epoch, prev_checkpointed_lr, _best_metric = \
                    checkpoint.initialize_params_from_file(
                        model=self.train_model,
                        weights_file=previous_checkpoint,
                        num_xpus=num_xpus,
                        opts=opts,
                        broadcast_computed_param=True,
                        reset_epoch=False,
                    )
        elif pretrained_model is not None and os.path.exists(pretrained_model):
            log.info("Load pretrained model: {}".format(pretrained_model))
            start_epoch, prev_checkpointed_lr, best_metric = \
                checkpoint.initialize_params_from_file(
                    model=self.train_model,
                    weights_file=pretrained_model,
                    num_xpus=num_xpus,
                    opts=opts,
                    broadcast_computed_param=True,
                    reset_epoch=opts['model_param']['reset_epoch'],
                )

        data_parallel_model.FinalizeAfterCheckpoint(self.train_model)

    def buildModelAndTrain(self, opts):
        log.info('in buildModelAndTrain, trainer_input: {}'.format(str(opts)))
        log.info("check type self: {}".format(type(self)))
        log.info("check self dir: {}".format(dir(self)))
        log.info("check self source: {}".format(self.__dict__))
        log.info("check self get_input_dataset methods: {}".
                 format(inspect.getsource(self.get_input_dataset)))
        log.info("check self gen_input_builder_fun method: {}".
                 format(inspect.getsource(self.gen_input_builder_fun)))
        log.info("check self gen_forward_pass_builder_fun method: {}".
                 format(inspect.getsource(self.gen_forward_pass_builder_fun)))
        if self.gen_param_update_builder_fun is not None:
            log.info("check self gen_param_update_builder_fun method: {}".
                     format(inspect.getsource(self.gen_param_update_builder_fun)))
        else:
            log.info("check self gen_optimizer_fun method: {}".
                     format(inspect.getsource(self.gen_optimizer_fun)))
        log.info("check self assembleAllOutputs method: {}".
                 format(inspect.getsource(self.assembleAllOutputs)))
        log.info("check self prep_data_parallel_models method: {}".
                 format(inspect.getsource(self.prep_data_parallel_models)))

        self.get_model_input_fun()

        self.init_model()

        self.planning_output()

        self.prep_data_parallel_models()

        self.loadCheckpoint()

        for epoch in self.list_of_epochs():

            log.info("start training epoch {}".format(epoch))

            self.fun_per_epoch_b4RunNet(epoch)

            for epoch_iter in self.list_of_epoch_iters():

                self.iter_start_time = time.time()

                self.fun_per_iter_b4RunNet(epoch, epoch_iter)

                if self.train_model is not None:
                    self.run_training_net()

                self.fun_per_iter_aftRunNetB4Test(epoch, epoch_iter)

                self.iter_end_time = time.time()

                if (epoch_iter %
                opts['epoch_iter']['num_train_iteration_per_test'] == 0):
                    secs_per_train = (self.iter_end_time - self.iter_start_time)
                    self.secs_per_train.append(secs_per_train)

                    sample_trained = self.total_batch_size
                    samples_per_sec = sample_trained / secs_per_train
                    self.samples_per_sec.append(samples_per_sec)

                    self.fract_epoch = (epoch +
                    float(epoch_iter) / self.epoch_iterations)
                    self.record_epochs.append(self.fract_epoch)

                    for key in self.metrics:
                        metric = self.metrics[key]
                        if not metric['is_train']:
                            continue
                        metric['calculator'].Add()
                        metric['output'].append(metric['calculator'].Compute())

                    self.test_loop_start_time = time.time()
                    for _test_iter in range(0, opts['epoch_iter']['num_test_iter']):
                        self.run_testing_net()
                        for key in self.metrics:
                            metric = self.metrics[key]
                            if metric['is_train']:
                                continue
                            metric['calculator'].Add()
                    self.test_loop_end_time = time.time()
                    self.sec_per_test_loop = \
                        self.test_loop_end_time - self.test_loop_start_time

                    for metric in self.metrics.values():
                        if metric['is_train']:
                            continue
                        metric['output'].append(metric['calculator'].Compute())

                    logStr = 'epoch:{}/{} iter:{}/{} secs_per_train:{} '.format(
                        self.fract_epoch, self.opts['epoch_iter']['num_epochs'],
                        epoch_iter, self.epoch_iterations, secs_per_train)
                    logStr += 'samples_per_sec:{} loop {} tests takes {} sec'.format(
                        samples_per_sec, opts['epoch_iter']['num_test_iter'],
                        self.sec_per_test_loop)
                    for metric, value in self.metrics.items():
                        logStr += ' {}:{} '.format(metric, value['output'][-1])
                    log.info('Iter Stats: {}'.format(logStr))

                self.fun_per_iter_aftRunNetAftTest(epoch, epoch_iter)

            self.checkpoint(epoch)

            self.fun_per_epoch_aftRunNet(epoch)

        self.fun_conclude_operator()

        self.createMetricsPlotsModelsOutputs()

        return self.assembleAllOutputs()
