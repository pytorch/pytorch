def basic_iteration_step(self, ddp_model, criterion, optimizer, hook_state, epoch, index, batch):
    r"""
    A function that performs an iteration of training.
    Args:
        ddp_model (nn.Module): distributed data parallel model
        criterion (nn.Module): loss function to measure model
        optimizer (optim.Optimizer): updates model parameters
        hook_state (object): ddp communication hook state object
        epoch (int): index of pass through the data
        index (int): iteration number - 1 in current batch
        batch (list): training examples
    """
    hook_state.next_batch()
    self.record_batch_start(self.epoch_key(epoch, index))
    optimizer.zero_grad()
    self.record_forward_start(self.epoch_key(epoch, index))
    loss = criterion(ddp_model(batch[0]), batch[1])
    self.record_forward_end(self.epoch_key(epoch, index))
    self.record_backward_start(self.epoch_key(epoch, index))
    loss.backward()
    self.record_backward_end(self.epoch_key(epoch, index))
    optimizer.step()
    self.record_batch_end(self.epoch_key(epoch, index))
