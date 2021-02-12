import torch.nn as nn
import torch
import argparse
import platform
import psutil
import os
import csv
import time
import pandas as pd
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __init__(self, shape, world_size, dtype = torch.float, lower_bound = 0, upper_bound = 1):
        self.len = shape[0] * world_size
        self.world_size = world_size
        if dtype == torch.float:
            self.data = torch.randn(*shape)
        elif dtype == torch.int:
            self.data = torch.randint(lower_bound, upper_bound, tuple(shape))   # fake input_ids

    def __getitem__(self, index):
        return self.data[int(index / self.world_size)]

    def __len__(self):
        return self.len


class LSTMLoader(DataLoader):
    def __init__(self, dataset, batch):
        super().__init__(dataset, batch_size=batch, shuffle=False)


# LSTM model, mocking classification procedure
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, layer=1, directions=1, num_class=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer = layer
        self.directions = directions == 2
        self.lstm = nn.LSTM(input_size, hidden_size, layer, bidirectional=self.directions)
        self.linear = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        self.lstm.flatten_parameters()  # for data parallel needs
        res, _ = self.lstm(x)
        res = res[:, -1, :]
        res = self.linear(res)
        return res

# hyperparameter
input_size = 224  # feature size
seq_len = 224  # sequence count

def train(precision="float"):
    # params
    batch = args.batch_size
    num_class = args.num_classes
    hidden_size = args.hidden_size
    layer = args.layer
    direction = args.direction

    benchmark = {}

    # training process
    loss_fn = nn.CrossEntropyLoss()
    target = torch.LongTensor(batch).random_(args.num_classes).cuda()    # uniform target to all training data
    model = LSTMModel(input_size, hidden_size, layer, direction, num_class)
    model = getattr(model, precision)().cuda()
    if args.distributed == True and args.dist_mode != 'horovod':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-5,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True)

    # if args.distributed == True and args.dist_mode == 'horovod':
    #     import horovod.torch as hvd
    #     optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=hvd.Compression.none, op=hvd.Average)
    #     hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    #     hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    duration = []
    print('Benchmarking Training LSTM precision type {} '.format(precision))
    for idx, data in enumerate(dataloader):
        start = time.time()
        data = getattr(data, precision)().cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        end = time.time()
        if idx >= args.warm_up:
            duration.append((end - start) * 1000)
    print('LSTM model average train time: {} ms'.format(sum(duration)/len(duration)))
    benchmark["LSTM_model"] = duration
    return benchmark, ' '.join(('LSTM',precision,'Train')), sum(duration)/len(duration)


def predict(precision="float"):
    num_class = args.num_classes    # number of classes to classify
    hidden_size = args.hidden_size
    layer = args.layer
    direction = args.direction

    model = LSTMModel(input_size, hidden_size, layer, direction, num_class)
    duration = []
    benchmark = {}
    print('Benchmarking predicting LSTM precision type {} '.format(precision))
    with torch.no_grad():
        model.eval()
        if args.num_gpu > 1:
            model = nn.DataParallel(model, device_ids=range(args.num_gpu))
        model = getattr(model, precision)().cuda()
        for idx, data in enumerate(dataloader):
            torch.cuda.synchronize()
            start = time.time()
            data = getattr(data, precision)().cuda()
            model(data)
            torch.cuda.synchronize()
            end = time.time()
            if idx >= args.warm_up:
                duration.append((end-start)*1000)
    print('LSTM model average predict time: {} ms'.format(sum(duration)/len(duration)))
    benchmark["LSTM_model"] = duration
    return benchmark, ' '.join(('LSTM',precision,'Predict')), sum(duration)/len(duration)

def Run(configs={}):
    # params config
    num_gpu = 1
    result_root = './Packages/ModelTrainingBenchmark/Pytorch_LSTMModel_Results/'
    os.makedirs(result_root, exist_ok=True)

    precisions = ["float", "half"]
    results = {}

    batch_size = 100
    if 'batch_size' in configs:
        batch_size = configs['batch_size']

    distributed = False
    if 'distributed' in configs:
        distributed = configs['distributed']

    dist_mode = 'horovod'
    if 'dist_mode' in configs:
        dist_mode = configs['dist_mode']

    parser = argparse.ArgumentParser(description='PyTorch Model Benchmark')
    parser.add_argument('--warm_up', '-w', type=int, default=8, required=False, help="Num of warm up")
    parser.add_argument('--num_test', '-n', type=int, default=256, required=False, help="Num of Test")
    parser.add_argument('--batch_size', '-b', type=int, default=batch_size, required=False, help='Num of batch size')
    parser.add_argument('--num_classes', '-c', type=int, default=100, required=False, help='Num of class')
    parser.add_argument('--num_gpu', '-g', type=int, default=num_gpu, required=False, help='Num of gpus')
    parser.add_argument('--hidden_size', '-hs', type=int, default=1000, required=False, help='Hidden size')
    parser.add_argument("--direction", "-d", type=int, choices=[1, 2], default=1, required=False, help='Single or Double direction')
    parser.add_argument("--layer", "-l", type=int, default=8, required=False, help='Stack layers')
    parser.add_argument('--local_rank', type=int, default=0, required=False, help='Local rank. Necessary for using the torch.distributed.launch utility.')
    parser.add_argument('--distributed', type=bool, default=distributed, required=False, help='Whether to enable distributed training.')
    parser.add_argument('--dist_mode', type=str, default=dist_mode, required=False, help='Distributed mode, such as horovod, native API.')

    global args
    args, unknown = parser.parse_known_args()
    print('Batch Size Per GPU : ' + str(args.batch_size))
    args.batch_size *= args.num_gpu

    world_size = 1
    # if args.distributed == True:
    #     if args.dist_mode == 'horovod':
    #         import horovod.torch as hvd
    #         hvd.init()
    #         world_size = int(hvd.size())
    #         if torch.cuda.is_available():
    #             torch.cuda.set_device(hvd.local_rank())
    #     else:
    #         dist.init_process_group(backend='nccl')
    #         world_size = int(os.environ['WORLD_SIZE'])
    #         if torch.cuda.is_available():
    #             torch.cuda.set_device(args.local_rank)

    # same data for training and predicting
    global dataloader
    samples_count = args.batch_size * (args.warm_up + args.num_test)
    train_dataset = RandomDataset([samples_count, input_size, seq_len], world_size, dtype=torch.float)
    train_sampler = None
    # if args.distributed == True:
    #     if args.dist_mode == 'horovod':
    #         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    #     else:
    #         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, sampler=train_sampler, drop_last=True, pin_memory=False)

    now = time.localtime()
    start_time = str(
        "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print('Time start : ', start_time)

    torch.backends.cudnn.benchmark = True
    device_name = str(torch.cuda.get_device_name(0))
    device_name = "".join((device_name, '_', str(args.num_gpu), '_gpus_'))

    system_configs = str(platform.uname())
    system_configs = '\n'.join((system_configs, str(psutil.cpu_freq()), 'cpu_count: ' + str(psutil.cpu_count()),
                                'memory_available: ' + str(psutil.virtual_memory().available)))

    gpu_configs = [torch.cuda.device_count(), torch.version.cuda, torch.backends.cudnn.version(),
                   torch.cuda.get_device_name(0)]
    gpu_configs = list(map(str, gpu_configs))

    temp = ['Number of GPUs on current device : ', 'CUDA Version : ', 'Cudnn Version : ', 'Device Name : ']
    for idx, value in enumerate(zip(temp, gpu_configs)):
        gpu_configs[idx] = ''.join(value)
        print(gpu_configs[idx])

    with open(os.path.join(result_root, "system_info.txt"), "w") as f:
        f.writelines('time start : ' + start_time + '\n')
        f.writelines('system_configs\n\n')
        f.writelines(system_configs)
        f.writelines('\ngpu_configs\n\n')
        f.writelines(s + '\n' for s in gpu_configs)

    print(system_configs)

    # launch the main procedure
    hidden_size = args.hidden_size
    layer = args.layer
    direction = args.direction

    print("LSTM model params: ")
    print("feature size: {}".format(input_size))
    print("sequence length: {}".format(seq_len))
    print("direction: {}".format(direction))
    print("hidden size: {}".format(hidden_size))
    print("stack layer: {}".format(layer))

    for precision in precisions:
        train_result, train_type, train_time = train(precision)
        train_result_df = pd.DataFrame(train_result)
        path = ''.join((result_root, '/', device_name, "_", precision, '_model_train_benchmark.csv'))
        train_result_df.to_csv(path, index=False)

        if args.distributed == True:
            continue

        inference_result, inference_type, inference_time = predict(precision)
        inference_result_df = pd.DataFrame(inference_result)
        path = ''.join((result_root, '/', device_name, "_", precision, '_model_prediction_benchmark.csv'))
        inference_result_df.to_csv(path, index=False)

        results[train_type] = [train_time]
        results[inference_type] = [inference_time]

    now = time.localtime()
    end_time = str(
        "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print('Time end : ', end_time)
    if args.distributed != True:
        # save run time result
        os.makedirs('./Results', exist_ok=True)
        runtime_result_path = os.path.join("./Results", "ModelTrainingBenchmark", "LSTM_runtime.csv")
        results = pd.DataFrame(results)
        results.to_csv(runtime_result_path, index=False)
if __name__ == '__main__':
    Run()
