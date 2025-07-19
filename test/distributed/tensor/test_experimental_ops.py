# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]


import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, Replicate
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


ITER_TIME = 10
LR = 0.001


class DistOtherOpsTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        # hard code world size to 2
        return 2

    @with_comms
    def test_slice(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Replicate()]

        input_list = torch.rand(ITER_TIME, 1024, 10)
        grad_output_list = torch.rand(ITER_TIME, 1024, 5) * 1e-3

        for i in range(ITER_TIME):
            inp = input_list[i].to(self.device_type).requires_grad_()
            grad_output = grad_output_list[i].to(self.device_type)

            # droppath  with dtensor
            inp_dtensor = distribute_tensor(inp, device_mesh, shard_spec)
            grad_output_dtensor = distribute_tensor(
                grad_output, device_mesh, shard_spec
            )
            output = inp_dtensor[:, :5]
            output.backward(grad_output_dtensor)

            # nll with plain tensor
            output_gt = inp[:, :5]
            output_gt.backward(grad_output)

            output_diff_abs = output.to_local() - output_gt
            output_diff_rel = output_diff_abs / (torch.abs(output_gt) + 1e-8)
            output_mse_abs = torch.mean(output_diff_abs * output_diff_abs).item()
            output_mse_rel = torch.mean(output_diff_rel * output_diff_rel).item()

            grad_diff_abs = inp_dtensor.grad.to_local() - inp.grad
            grad_diff_rel = grad_diff_abs / (torch.abs(inp.grad) + 1e-8)
            grad_mse_abs = torch.mean(grad_diff_abs * grad_diff_abs).item()
            grad_mse_rel = torch.mean(grad_diff_rel * grad_diff_rel).item()

            self.assertTrue(
                output_mse_abs <= 1e-6,
                f"Too large absolute mse for output, expected less equal 1e-6, got {output_mse_abs}",
            )
            self.assertTrue(
                output_mse_rel <= 1e-6,
                f"Too large relative mse for output, expected less equal 1e-6, got {output_mse_rel}",
            )
            self.assertTrue(
                grad_mse_abs <= 1e-6,
                f"Too large absolute mse for gradient, expected less equal 1e-6, got {grad_mse_abs}",
            )
            self.assertTrue(
                grad_mse_rel <= 1e-6,
                f"Too large relative mse for gradient, expected less equal 1e-6, got {grad_mse_rel}",
            )

    @with_comms
    def test_bernoulli(self):
        rank = dist.get_rank()
        device_mesh = self.build_device_mesh()
        shard_spec = [Replicate()]

        input_list = torch.rand(ITER_TIME, 1024, 10)
        grad_output_list = torch.rand(ITER_TIME, 1024, 10) * 1e-3

        for i in range(ITER_TIME):
            inp = input_list[i].to(self.device_type).requires_grad_()
            grad_output = grad_output_list[i].to(self.device_type)

            # bernoulli  with dtensor
            inp_dtensor = distribute_tensor(inp, device_mesh, shard_spec)
            grad_output_dtensor = distribute_tensor(
                grad_output, device_mesh, shard_spec
            )
            output = torch.bernoulli(inp_dtensor)
            output.backward(grad_output_dtensor)

            send_output_tensor = output.to_local()
            recv_output_tensor = torch.zeros_like(send_output_tensor)

            send_grad_tensor = inp_dtensor.grad.to_local()
            recv_grad_tensor = torch.zeros_like(send_grad_tensor)

            send_op_1 = dist.P2POp(dist.isend, send_output_tensor, 1 ^ rank)
            send_op_2 = dist.P2POp(dist.isend, send_grad_tensor, 1 ^ rank)
            recv_op_1 = dist.P2POp(dist.irecv, recv_output_tensor, 1 ^ rank)
            recv_op_2 = dist.P2POp(dist.irecv, recv_grad_tensor, 1 ^ rank)

            reqs = dist.batch_isend_irecv([send_op_1, send_op_2, recv_op_1, recv_op_2])
            for req in reqs:
                req.wait()

            output_diff_abs = send_output_tensor - recv_output_tensor
            output_diff_rel = output_diff_abs / (torch.abs(recv_output_tensor) + 1e-8)
            output_mse_abs = torch.mean(output_diff_abs * output_diff_abs).item()
            output_mse_rel = torch.mean(output_diff_rel * output_diff_rel).item()

            grad_diff_abs = send_grad_tensor - recv_grad_tensor
            grad_diff_rel = grad_diff_abs / (torch.abs(recv_grad_tensor) + 1e-8)
            grad_mse_abs = torch.mean(grad_diff_abs * grad_diff_abs).item()
            grad_mse_rel = torch.mean(grad_diff_rel * grad_diff_rel).item()

            self.assertTrue(
                output_mse_abs <= 1e-6,
                f"Too large absolute mse for output, expected less equal 1e-6, got {output_mse_abs}",
            )
            self.assertTrue(
                output_mse_rel <= 1e-6,
                f"Too large relative mse for output, expected less equal 1e-6, got {output_mse_rel}",
            )
            self.assertTrue(
                grad_mse_abs <= 1e-6,
                f"Too large absolute mse for gradient, expected less equal 1e-6, got {grad_mse_abs}",
            )
            self.assertTrue(
                grad_mse_rel <= 1e-6,
                f"Too large relative mse for gradient, expected less equal 1e-6, got {grad_mse_rel}",
            )

    @with_comms
    def test_nll(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Replicate()]

        pred_list = torch.rand(ITER_TIME, 1024, 10)
        target_list = torch.randint(0, 10, (ITER_TIME, 1024), dtype=torch.long)

        criterion = torch.nn.CrossEntropyLoss()

        for i in range(ITER_TIME):
            pred = pred_list[i].to(self.device_type).requires_grad_()
            target = target_list[i].to(self.device_type)

            # nll with dtensor
            pred_dtensor = distribute_tensor(pred, device_mesh, shard_spec)
            target_dtensor = distribute_tensor(target, device_mesh, shard_spec)
            loss = criterion(pred_dtensor, target_dtensor)
            loss.backward()

            # nll with plain tensor
            loss_gt = criterion(pred, target)
            loss_gt.backward()

            loss_diff_abs = loss.to_local() - loss_gt
            loss_diff_rel = loss_diff_abs / (torch.abs(loss_gt) + 1e-8)
            loss_mse_abs = torch.mean(loss_diff_abs * loss_diff_abs).item()
            loss_mse_rel = torch.mean(loss_diff_rel * loss_diff_rel).item()

            grad_diff_abs = pred_dtensor.grad.to_local() - pred.grad
            grad_diff_rel = grad_diff_abs / (torch.abs(pred.grad) + 1e-8)
            grad_mse_abs = torch.mean(grad_diff_abs * grad_diff_abs).item()
            grad_mse_rel = torch.mean(grad_diff_rel * grad_diff_rel).item()

            self.assertTrue(
                loss_mse_abs <= 1e-6,
                f"Too large absolute mse for loss, expected less equal 1e-6, got {loss_mse_abs}",
            )
            self.assertTrue(
                loss_mse_rel <= 1e-6,
                f"Too large relative mse for loss, expected less equal 1e-6, got {loss_mse_rel}",
            )
            self.assertTrue(
                grad_mse_abs <= 1e-6,
                f"Too large absolute mse for gradient, expected less equal 1e-6, got {grad_mse_abs}",
            )
            self.assertTrue(
                grad_mse_rel <= 1e-6,
                f"Too large relative mse for gradient, expected less equal 1e-6, got {grad_mse_rel}",
            )


if __name__ == "__main__":
    run_tests()
