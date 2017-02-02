from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace, cnn, utils
from caffe2.python.recurrent import LSTM

import argparse
import logging
import numpy as np


logging.basicConfig()
log = logging.getLogger("char_rnn")
log.setLevel(logging.DEBUG)


# Default set() here is intentional as it would accumulate values like a global
# variable
def CreateNetOnce(net, created_names=set()):
    name = net.Name()
    if name not in created_names:
        created_names.add(name)
        workspace.CreateNet(net)


class CharRNN(object):
    seq_length = 25
    batch_size = 1
    iters_to_report = 500
    hidden_size = 100

    def __init__(self, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        self.iters_to_report = args.iters_to_report
        self.hidden_size = args.hidden_size

        with open(args.train_data) as f:
            self.text = f.read()

        self.vocab = list(set(self.text))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}
        self.D = len(self.char_to_idx)

        print("Input has {} characters. Total input size: {}".format(
            len(self.vocab), len(self.text)))

    def CreateModel(self):
        log.debug("Start training")
        model = cnn.CNNModelHelper(name="char_rnn")

        input_blob, seq_lengths, hidden_init, cell_init, target = \
            model.net.AddExternalInputs(
                'input_blob',
                'seq_lengths',
                'hidden_init',
                'cell_init',
                'target',
            )

        hidden_output_all, self.hidden_output, _, self.cell_state = LSTM(
            model, input_blob, seq_lengths, (hidden_init, cell_init),
            self.D, self.hidden_size, scope="LSTM")
        output = model.FC(hidden_output_all, None, dim_in=self.hidden_size,
                          dim_out=self.D, axis=2)

        # axis is 2 as first two are T (time) and N (batch size).
        # We treat them as one big batch of size T * N
        softmax = model.Softmax(output, 'softmax', axis=2)

        softmax_reshaped, _ = model.Reshape(
            softmax, ['softmax_reshaped', '_'], shape=[-1, self.D])

        # Create a copy of the current net. We will use it on the forward
        # pass where we don't need loss and backward operators
        self.forward_net = core.Net(model.net.Proto())

        xent = model.LabelCrossEntropy([softmax_reshaped, target], 'xent')
        loss = model.ReduceFrontSum(xent, 'loss')
        model.AddGradientOperators([loss])

        ITER = model.Iter("iter")
        LR = model.LearningRate(
            ITER, "LR", base_lr=-0.1 / self.batch_size,
            policy="step", stepsize=1, gamma=0.9999)
        ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)

        for param in model.params:
            param_grad = model.param_to_grad[param]
            model.WeightedSum([param, ONE, param_grad, LR], param)

        self.model = model
        self.predictions = softmax
        self.loss = loss

        self.prepare_state = core.Net("prepare_state")
        self.prepare_state.Copy(self.hidden_output, hidden_init)
        self.prepare_state.Copy(self.cell_state, cell_init)

    def _idx_at_pos(self, pos):
        return self.char_to_idx[self.text[pos]]

    def TrainModel(self):
        log.debug("Training model")

        workspace.RunNetOnce(self.model.param_init_net)

        # As though we predict the same probablity for each character
        smooth_loss = -np.log(1.0 / self.D) * self.seq_length
        last_n_iter = 0
        last_n_loss = 0.0
        pos, num_iter = 0, 0
        N = len(self.text)

        # Writing to output states which will be copied to input
        # states within the loop below
        workspace.FeedBlob(self.hidden_output, np.zeros(
            [1, self.batch_size, self.hidden_size], dtype=np.float32
        ))
        workspace.FeedBlob(self.cell_state, np.zeros(
            [1, self.batch_size, self.hidden_size], dtype=np.float32
        ))
        workspace.CreateNet(self.prepare_state)

        # We iterate over text in a loop many times. Each time we peak
        # seq_length segment and feed it to LSTM as a sequence
        while True:
            workspace.FeedBlob(
                "seq_lengths", np.array([self.seq_length]).astype(np.int32))
            workspace.RunNet(self.prepare_state.Name())

            input = np.zeros(
                [self.seq_length, self.batch_size, self.D]
            ).astype(np.float32)
            target = np.zeros(
                [self.seq_length * self.batch_size]
            ).astype(np.int32)

            for e in range(self.batch_size):
                for i in range(self.seq_length):
                    input[i][e][self._idx_at_pos(pos)] = 1
                    target[i * self.batch_size + e] =\
                        self._idx_at_pos((pos + 1) % N)
                    pos = (pos + 1) % N

            workspace.FeedBlob('input_blob', input)
            workspace.FeedBlob('target', target)

            CreateNetOnce(self.model.net)
            workspace.RunNet(self.model.net.Name())

            num_iter += 1
            last_n_iter += 1

            if num_iter % self.iters_to_report == 0:
                print("{} Iteration {} {}".
                      format('-' * 10, num_iter, '-' * 10))

            loss = workspace.FetchBlob(self.loss) / self.batch_size
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            last_n_loss += loss

            if num_iter % self.iters_to_report == 0:
                self.GenerateText(500, np.random.choice(self.vocab))

                log.debug("Loss since last report: {}"
                          .format(last_n_loss / last_n_iter))
                log.debug("Smooth loss: {}".format(smooth_loss))

                last_n_loss = 0.0
                last_n_iter = 0

    def GenerateText(self, num_characters, ch):
        # Given a starting symbol we feed a fake sequence of size 1 to
        # our RNN num_character times. After each time we use output
        # probabilities to pick a next character to feed to the network.
        # Same character becomes part of the output
        CreateNetOnce(self.forward_net)

        text = '' + ch
        for i in range(num_characters):
            workspace.FeedBlob(
                "seq_lengths", np.array([1]).astype(np.int32))
            workspace.RunNet(self.prepare_state.Name())

            input = np.zeros([1, self.batch_size, self.D]).astype(np.float32)
            input[0][0][self.char_to_idx[ch]] = 1
            workspace.FeedBlob("input_blob", input)

            workspace.RunNet(self.forward_net.Name())

            p = workspace.FetchBlob(self.predictions)
            next = np.random.choice(self.D, p=p[0][0])

            ch = self.idx_to_char[next]
            text += ch

        print(text)


@utils.debug
def main():
    parser = argparse.ArgumentParser(
        description="Caffe2: Char RNN Training"
    )
    parser.add_argument("--train_data", type=str, default=None,
                        help="Path to training data in a text file format",
                        required=True)
    parser.add_argument("--seq_length", type=int, default=25,
                        help="One training example sequence length")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--iters_to_report", type=int, default=500,
                        help="How often to report loss and generate text")
    parser.add_argument("--hidden_size", type=int, default=100,
                        help="Dimention of the hidden representation")

    args = parser.parse_args()

    model = CharRNN(args)
    model.CreateModel()
    model.TrainModel()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
