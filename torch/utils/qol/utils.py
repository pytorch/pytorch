"""
Utility functions for PyTorch - Framework-agnostic "missing" functions

Provides:
- lazy_flatten: Auto-shape flattener for transitioning from Conv to Linear layers
- loss_ncc: Normalized Cross-Correlation loss (for medical imaging)
- find_lr: Learning Rate Finder (LR range test)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import onnx
import onnxruntime as ort


def lazy_flatten(tensor: torch.Tensor, start_dim: int = 1) -> torch.Tensor:
    """
    Auto-shape flattener that automatically calculates correct dimensions.
    
    Eliminates the need to manually calculate flattened sizes when transitioning
    from convolutional layers to fully connected layers.
    
    Args:
        tensor: Input tensor to flatten
        start_dim: Dimension to start flattening from (default: 1, preserves batch)
    
    Returns:
        Flattened tensor
    
    Example:
        >>> x = torch.randn(32, 16, 7, 7)  # batch=32, channels=16, H=7, W=7
        >>> x_flat = lazy_flatten(x)       # Shape: [32, 784]
        >>> # No need to calculate 16*7*7 manually!
    """
    return torch.flatten(tensor, start_dim=start_dim)


def get_flatten_size(input_shape: Tuple[int, ...]) -> int:
    """
    Calculate the flattened size given an input shape.
    
    Useful for determining the input size of the first Linear layer after Conv layers.
    
    Args:
        input_shape: Shape of the tensor (excluding batch dimension)
                    e.g., (16, 7, 7) for Conv output
    
    Returns:
        Total flattened size
    
    Example:
        >>> size = get_flatten_size((16, 7, 7))
        >>> print(size)  # 784
        >>> # Now you can use: nn.Linear(size, 128)
    """
    return np.prod(input_shape)


def loss_ncc(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalized Cross-Correlation (NCC) loss function.
    
    Commonly used in medical image registration and similarity measurement.
    NCC is robust to intensity variations and widely used in ACDC, ADNI datasets.
    
    NCC = 1 - [ sum((y_true - mean_true) * (y_pred - mean_pred)) / 
                (sqrt(sum((y_true - mean_true)^2)) * sqrt(sum((y_pred - mean_pred)^2))) ]
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        eps: Small constant for numerical stability (default: 1e-8)
    
    Returns:
        NCC loss value (lower is better, range: 0 to 2)
    
    Example:
        >>> y_true = torch.randn(8, 1, 64, 64)
        >>> y_pred = model(x)
        >>> loss = loss_ncc(y_true, y_pred)
        >>> loss.backward()
    
    Note:
        - Returns 0 when images are identical
        - Returns 2 when images are completely anti-correlated
        - More robust than MSE for images with intensity variations
    """
    # Flatten spatial dimensions
    y_true_flat = y_true.view(y_true.size(0), -1)
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    
    # Calculate means
    mean_true = torch.mean(y_true_flat, dim=1, keepdim=True)
    mean_pred = torch.mean(y_pred_flat, dim=1, keepdim=True)
    
    # Center the tensors
    y_true_centered = y_true_flat - mean_true
    y_pred_centered = y_pred_flat - mean_pred
    
    # Calculate cross-correlation
    numerator = torch.sum(y_true_centered * y_pred_centered, dim=1)
    
    # Calculate standard deviations
    std_true = torch.sqrt(torch.sum(y_true_centered ** 2, dim=1) + eps)
    std_pred = torch.sqrt(torch.sum(y_pred_centered ** 2, dim=1) + eps)
    
    # NCC coefficient
    ncc = numerator / (std_true * std_pred + eps)
    
    # Return as loss (1 - NCC), averaged over batch
    return torch.mean(1.0 - ncc)


def ncc_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalized Cross-Correlation score (similarity metric).
    
    Returns the NCC coefficient itself (higher is better, range: -1 to 1).
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        eps: Small constant for numerical stability
    
    Returns:
        NCC score (higher is better, 1 = perfect match)
    """
    return 1.0 - loss_ncc(y_true, y_pred, eps)


class LRFinder:
    """
    Learning Rate Finder for PyTorch models.
    
    Implements the LR range test popularized by fast.ai to find optimal learning rates.
    Runs the model for a few iterations with exponentially increasing learning rates
    and plots the loss to identify the "knee" of the curve.
    
    Example:
        >>> lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
        >>> lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100)
        >>> lr_finder.plot()
        >>> best_lr = lr_finder.get_best_lr()
        >>> print(f"Suggested LR: {best_lr}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: str = 'cpu'
    ):
        """
        Initialize LR Finder.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance (will be modified during search)
            criterion: Loss function
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Store original state
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())
        
        # Results
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.best_lr: Optional[float] = None
    
    def range_test(
        self,
        dataloader: torch.utils.data.DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0
    ):
        """
        Perform LR range test.
        
        Args:
            dataloader: Training data loader
            start_lr: Starting learning rate (default: 1e-7)
            end_lr: Ending learning rate (default: 10.0)
            num_iter: Number of iterations to run (default: 100)
            smooth_f: Loss smoothing factor (default: 0.05)
            diverge_th: Stop if loss > diverge_th * best_loss (default: 5.0)
        """
        # Reset to original state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        self.model.to(self.device)
        self.model.train()
        
        # Calculate LR multiplier
        lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
        
        # Initialize
        lr = start_lr
        self._set_lr(lr)
        
        avg_loss = 0.0
        best_loss = float('inf')
        batch_iter = iter(dataloader)
        
        self.lrs = []
        self.losses = []
        
        with tqdm(range(num_iter), desc="LR Finder") as pbar:
            for iteration in pbar:
                # Get batch
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(dataloader)
                    batch = next(batch_iter)
                
                # Unpack batch
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Calculate loss
                if targets is not None:
                    loss = self.criterion(outputs, targets)
                else:
                    loss = self.criterion(outputs)
                
                # Check for nan/inf
                if torch.isnan(loss) or torch.isinf(loss):
                    break
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Smooth loss
                loss_val = loss.item()
                if iteration == 0:
                    avg_loss = loss_val
                else:
                    avg_loss = smooth_f * loss_val + (1 - smooth_f) * avg_loss
                
                # Record
                self.lrs.append(lr)
                self.losses.append(avg_loss)
                
                # Update best loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                
                # Check for divergence
                if avg_loss > diverge_th * best_loss:
                    print(f"\nStopping early - loss diverged at LR={lr:.2e}")
                    break
                
                # Update progress bar
                pbar.set_postfix({'lr': f'{lr:.2e}', 'loss': f'{avg_loss:.4f}'})
                
                # Increase learning rate
                lr *= lr_mult
                self._set_lr(lr)
        
        # Restore original state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        # Calculate best LR
        self.best_lr = self._calculate_best_lr()
    
    def _set_lr(self, lr: float):
        """Set learning rate for all param groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _calculate_best_lr(self) -> Optional[float]:
        """
        Calculate the best learning rate based on the steepest gradient.
        
        Returns the LR where the loss decrease is steepest (before divergence).
        """
        if len(self.losses) < 2:
            return None
        
        # Calculate gradients of loss
        losses_array = np.array(self.losses)
        lrs_array = np.array(self.lrs)
        
        # Find minimum loss and its index
        min_loss_idx = np.argmin(losses_array)
        
        # Calculate gradient (rate of change)
        gradients = np.gradient(losses_array)
        
        # Find steepest negative gradient before minimum
        # (where loss is decreasing fastest)
        if min_loss_idx > 10:
            steepest_idx = np.argmin(gradients[:min_loss_idx])
            return lrs_array[steepest_idx]
        else:
            # Return LR at 1/10th of minimum if not enough data
            return lrs_array[min_loss_idx] / 10.0
    
    def plot(
        self,
        skip_start: int = 10,
        skip_end: int = 5,
        log_lr: bool = True,
        show_best: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Plot the learning rate vs loss curve.
        
        Args:
            skip_start: Skip first N points (default: 10)
            skip_end: Skip last N points (default: 5)
            log_lr: Use log scale for LR axis (default: True)
            show_best: Mark the suggested best LR (default: True)
            save_path: Path to save the plot (optional)
        """
        if len(self.lrs) == 0:
            print("No data to plot. Run range_test() first.")
            return
        
        # Prepare data
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses, linewidth=2)
        
        if log_lr:
            plt.xscale('log')
        
        # Mark best LR
        if show_best and self.best_lr is not None:
            plt.axvline(x=self.best_lr, color='r', linestyle='--', 
                       label=f'Suggested LR: {self.best_lr:.2e}')
            plt.legend()
        
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Learning Rate Finder', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def get_best_lr(self) -> Optional[float]:
        """
        Get the suggested best learning rate.
        
        Returns:
            Suggested learning rate or None if range_test not run
        """
        return self.best_lr


def find_lr(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iter: int = 100,
    plot: bool = True
) -> float:
    """
    Convenience function to find optimal learning rate.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        criterion: Loss function
        dataloader: Training data loader
        device: Device to run on ('cpu' or 'cuda')
        start_lr: Starting learning rate (default: 1e-7)
        end_lr: Ending learning rate (default: 10.0)
        num_iter: Number of iterations (default: 100)
        plot: Whether to show plot (default: True)
    
    Returns:
        Suggested optimal learning rate
    
    Example:
        >>> best_lr = find_lr(model, optimizer, nn.CrossEntropyLoss(), 
        ...                   train_loader, device='cuda')
        >>> print(f"Use learning rate: {best_lr}")
        >>> # Now create optimizer with best_lr
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    """
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(dataloader, start_lr, end_lr, num_iter)
    
    if plot:
        lr_finder.plot()
    
    best_lr = lr_finder.get_best_lr()
    
    if best_lr is None:
        print("Warning: Could not determine best LR. Using default 1e-3")
        return 1e-3
    
    return best_lr


def toONNX(
    model: nn.Module,
    dummy_input,
    filepath: str,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    opset_version: int = 11,
    dynamic_axes: Optional[dict] = None,
    verbose: bool = False,
):
    """
    Export a PyTorch `nn.Module` to ONNX format.

    Args:
        model: PyTorch model to export
        dummy_input: example input (torch.Tensor or tuple of tensors)
        filepath: output path for the .onnx file
        input_names: optional list of input names
        output_names: optional list of output names
        opset_version: ONNX opset version
        dynamic_axes: dynamic axes mapping for inputs/outputs
        verbose: pass verbose flag to torch.onnx.export

    Returns:
        The filepath where the ONNX model was saved.
    """
    model_cpu = model
    was_training = model_cpu.training
    model_cpu.eval()

    # Default names
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    # Export
    torch.onnx.export(
        model_cpu,
        dummy_input,
        filepath,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )

    # restore training state
    model_cpu.train(was_training)
    return filepath


def fromONNX(onnx_path: str, inputs, providers: Optional[list] = None):
    """
    Run inference on an ONNX model using onnxruntime.

    Args:
        onnx_path: path to the ONNX model file
        inputs: single input (tensor/ndarray) or dict mapping input names to data
        providers: optional list of ONNXRuntime providers

    Returns:
        List of numpy outputs from the ONNX model (or single ndarray if one output)
    """
    sess_options = ort.SessionOptions()
    if providers is None:
        # default providers order
        providers = None

    # Use only available providers to avoid noisy warnings when optional providers
    # are not installed in the environment.
    available = ort.get_available_providers()
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=(providers if providers is not None else available) or None,
    )

    # Build input dict
    input_map = {}
    ort_inputs = sess.get_inputs()
    if isinstance(inputs, dict):
        for k, v in inputs.items():
            if hasattr(v, 'cpu'):
                v = v.cpu().detach().numpy()
            input_map[k] = np.asarray(v)
    else:
        # single input -> map to first input name
        v = inputs
        if hasattr(v, 'cpu'):
            v = v.cpu().detach().numpy()
        input_map[ort_inputs[0].name] = np.asarray(v)

    out = sess.run(None, input_map)
    if len(out) == 1:
        return out[0]
    return out
