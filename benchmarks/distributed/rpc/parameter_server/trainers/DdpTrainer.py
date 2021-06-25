import torch

from .TrainerBase import TrainerBase


class DdpTrainer(TrainerBase):

    def __init__(
        self,
        rank,
        trainer_count,
        process_group,
        use_cuda_rpc,
        server_rref,
        backend, epochs,
        preprocess_data,
        create_criterion,
        create_ddp_model,
        lr,
        create_optimizer,
        HookState,
        hook,
        iteration_step
    ):
        r"""
        A trainer that implements a DDP training algorithm using a simple hook that performs allreduce
        using the process_group implementation.
        Args:
            rank (int): worker rank
            trainer_count (int): count of trainer in the world
            process_group (ProcessGroup): distributed process group
            use_cuda_rpc (bool): indicator for CUDA RPC
            server_rref (RRef): remote reference to the server
            backend (str): distributed communication backend
            epochs (int): epoch count for training
            preprocess_data (function): preprocesses data passed
                to the trainer before starting training
            create_criterion (function): creates a criterion to calculate loss
            create_ddp_model (function): creates a ddp model for the trainer
            lr (float): learning rate used by the optimizer during training
            create_optimizer (function): creates the optimizer that will
                update the gradients each iteration step
            HookState (class): class that will be used to keep tracking of state
                during training.
            hook (function): ddp communication hook
            iteration_step (function): will perform 1 step of training
        """
        super().__init__(rank)
        self.rank = rank
        self.trainer_count = trainer_count
        self.process_group = process_group
        self.use_cuda_rpc = use_cuda_rpc
        self.server_rref = server_rref
        self.backend = backend
        self.epochs = epochs
        self.preprocess_data = preprocess_data
        self.create_criterion = create_criterion
        self.create_ddp_model = create_ddp_model
        self.lr = lr
        self.create_optimizer = create_optimizer
        self.HookState = HookState
        self.hook = hook
        self.iteration_step = iteration_step

    def epoch_key(self, epoch, index):
        r"""
        A method that returns an encoded key that represents the current epoch and
        iteration index.
        Args:
            epoch (int): epoch index
            index (int): iteration index
        """
        return f"{epoch},{index}"

    def train(self, model, data):
        r"""
        A method that implements the training algorithm.
        Args:
            model (nn.Module): neural network model
            data (list): training examples
        """
        model = model.cuda(self.rank)
        data = self.preprocess_data(self.rank, data)
        criterion = self.create_criterion(self.rank, model)
        ddp_model, hook_state = self.create_ddp_model(
            self, self.rank, model, self.process_group, self.HookState, self.hook
        )
        optimizer = self.create_optimizer(ddp_model.parameters(), self.lr)

        for epoch in range(self.epochs):
            if epoch % 5 == 0 and self.rank == 0:
                print(f"train epoch={epoch}")
            for index, batch in enumerate(data):
                self.iteration_step(
                    self, ddp_model, criterion, optimizer, hook_state, epoch, index, batch
                )
        torch.cuda.synchronize(self.rank)
