#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
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

"""
This example shows how to use higher to do Model Agnostic Meta Learning (MAML)
for few-shot Omniglot classification.
For more details see the original MAML paper:
https://arxiv.org/abs/1703.03400

This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py

Our MAML++ fork and experiments are available at:
https://github.com/bamos/HowToTrainYourMAMLPytorch
"""

from support.omniglot_loaders import OmniglotNShot
from functorch import make_functional_with_buffers
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import torch
import matplotlib.pyplot as plt
import argparse
import time

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
plt.style.use('bmh')


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n-way', '--n_way', type=int, help='n way', default=5)
    argparser.add_argument(
        '--k-spt', '--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument(
        '--k-qry', '--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument(
        '--device', type=str, help='device', default='cuda')
    argparser.add_argument(
        '--task-num', '--task_num',
        type=int,
        help='meta batch size, namely task num',
        default=32)
    argparser.add_argument('--seed', type=int, help='random seed', default=1)
    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Set up the Omniglot loader.
    device = args.device
    db = OmniglotNShot(
        '/tmp/omniglot-data',
        batchsz=args.task_num,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        imgsz=28,
        device=device,
    )

    # Create a vanilla PyTorch neural network that will be
    # automatically monkey-patched by higher later.
    # Before higher, models could *not* be created like this
    # and the parameters needed to be manually updated and copied
    # for the updates.
    net = nn.Sequential(
        nn.Conv2d(1, 64, 3),
        nn.BatchNorm2d(64, momentum=1, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, momentum=1, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, momentum=1, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        Flatten(),
        nn.Linear(64, args.n_way)).to(device)

    net.train()
    fnet, params, buffers = make_functional_with_buffers(net)

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_opt = optim.Adam(params, lr=1e-3)

    log = []
    for epoch in range(100):
        train(db, [params, buffers, fnet], device, meta_opt, epoch, log)
        test(db, [params, buffers, fnet], device, epoch, log)
        plot(log)


def train(db, net, device, meta_opt, epoch, log):
    params, buffers, fnet = net
    n_train_iter = db.x_train.shape[0] // db.batchsz

    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db.next()

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        n_inner_iter = 5
        # inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()
        for i in range(task_num):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.
            new_params = params
            for _ in range(n_inner_iter):
                spt_logits = fnet(new_params, buffers, x_spt[i])
                spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                grads = torch.autograd.grad(spt_loss, new_params, create_graph=True)
                new_params = [p - g * 1e-1 for p, g, in zip(new_params, grads)]

            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            qry_logits = fnet(new_params, buffers, x_qry[i])
            qry_loss = F.cross_entropy(qry_logits, y_qry[i])
            qry_losses.append(qry_loss.detach())
            qry_acc = (qry_logits.argmax(
                dim=1) == y_qry[i]).sum().item() / querysz
            qry_accs.append(qry_acc)

            # Update the model's meta-parameters to optimize the query
            # losses across all of the tasks sampled in this batch.
            # This unrolls through the gradient steps.
            qry_loss.backward()

        meta_opt.step()
        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100. * sum(qry_accs) / task_num
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time
        if batch_idx % 4 == 0:
            print(
                f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
            )

        log.append({
            'epoch': i,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'train',
            'time': time.time(),
        })


def test(db, net, device, epoch, log):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    [params, buffers, fnet] = net
    n_test_iter = db.x_test.shape[0] // db.batchsz

    qry_losses = []
    qry_accs = []

    for batch_idx in range(n_test_iter):
        x_spt, y_spt, x_qry, y_qry = db.next('test')
        task_num, setsz, c_, h, w = x_spt.size()

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?
        n_inner_iter = 5

        for i in range(task_num):
            new_params = params
            for _ in range(n_inner_iter):
                spt_logits = fnet(new_params, buffers, x_spt[i])
                spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                grads = torch.autograd.grad(spt_loss, new_params)
                new_params = [p - g * 1e-1 for p, g, in zip(new_params, grads)]

            # The query loss and acc induced by these parameters.
            qry_logits = fnet(new_params, buffers, x_qry[i]).detach()
            qry_loss = F.cross_entropy(
                qry_logits, y_qry[i], reduction='none')
            qry_losses.append(qry_loss.detach())
            qry_accs.append(
                (qry_logits.argmax(dim=1) == y_qry[i]).detach())

    qry_losses = torch.cat(qry_losses).mean().item()
    qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
    print(
        f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}'
    )
    log.append({
        'epoch': epoch + 1,
        'loss': qry_losses,
        'acc': qry_accs,
        'mode': 'test',
        'time': time.time(),
    })


def plot(log):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'maml-accs.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)


# Won't need this after this PR is merged in:
# https://github.com/pytorch/pytorch/pull/22245
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


if __name__ == '__main__':
    main()
