"""
SmartSummary - Advanced model summary with bottleneck detection

Provides comprehensive model analysis including:
- Layer-by-layer parameter counts and shapes
- Gradient variance analysis
- Bottleneck layer identification
- Memory usage estimation
- FLOPS estimation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import numpy as np


class SmartSummary:
    """
    Advanced model summary that identifies bottlenecks and provides detailed insights.
    
    Features beyond standard model.summary():
    - Gradient variance tracking to identify unstable layers
    - Bottleneck detection based on parameters and computational cost
    - Memory usage estimation
    - Output shape inference for complex models
    - Pretty-printed table format
    
    Example:
        >>> model = YourModel()
        >>> summary = SmartSummary(model, input_size=(3, 224, 224))
        >>> summary.show()
        >>> bottlenecks = summary.get_bottlenecks()
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        input_size: Optional[Tuple[int, ...]] = None,
        batch_size: int = 1,
        device: str = "cpu",
        track_gradients: bool = False,
        # thresholds/config
        grad_large_threshold: float = 1e3,
        grad_zero_tol: float = 0.0,
        init_std_warn_multiply: float = 10.0,
        init_std_warn_min_mult: float = 0.01,
        param_ratio_bottleneck: float = 0.1,
        activation_bottleneck_mb: float = 10.0,
        keep_activations: bool = False
    ):
        """
        Initialize SmartSummary.
        
        Args:
            model: PyTorch model to analyze
            input_size: Input tensor size (excluding batch dimension)
                       e.g., (3, 224, 224) for images
            batch_size: Batch size for shape inference
            device: Device to run analysis on ('cpu' or 'cuda')
            track_gradients: Whether to track gradient statistics (requires forward+backward pass)
        """
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = device
        self.track_gradients = track_gradients
        
        self.summary_data: OrderedDict = OrderedDict()
        self.total_params = 0
        self.trainable_params = 0
        self.total_output_size = 0
        self.gradient_stats: Dict[str, Dict[str, float]] = {}
        self.init_warnings: Dict[str, str] = {}
        self.receptive_field: Dict[str, Dict[str, float]] = {}
        self.memory_profile: Dict[str, Any] = {}
        # thresholds
        self.grad_large_threshold = grad_large_threshold
        self.grad_zero_tol = grad_zero_tol
        self.init_std_warn_multiply = init_std_warn_multiply
        self.init_std_warn_min_mult = init_std_warn_min_mult
        self.param_ratio_bottleneck = param_ratio_bottleneck
        self.activation_bottleneck_mb = activation_bottleneck_mb
        self.keep_activations = keep_activations
        # option to keep strong refs (may increase memory) and limit
        self.keep_activations_strong: bool = False
        self.max_saved_activation_bytes: Optional[int] = None

        # runtime maps
        self.tensor_rf_map: Dict[int, Dict[str, Any]] = {}
        self.saved_activation_refs: Dict[str, Any] = {}
        
        self._analyze_model()
    
    def _register_hooks(self, module_hooks: List):
        """Register forward hooks to capture layer information"""
        def hook_fn(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(self.summary_data)
            
            m_key = f"{class_name}-{module_idx}"
            
            # Calculate output shape
            if isinstance(output, (list, tuple)):
                output_shape = [list(o.size()) for o in output if isinstance(o, torch.Tensor)]
            else:
                output_shape = list(output.size())
            
            # Count parameters
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            # Estimate output size in MB
            if isinstance(output, (list, tuple)):
                output_size = sum(o.numel() * o.element_size() for o in output if isinstance(o, torch.Tensor))
            else:
                output_size = output.numel() * output.element_size()

            # activation bytes and optional weakref saving
            try:
                act_bytes = int(output_size)
            except Exception:
                act_bytes = 0

            if self.keep_activations:
                # store activation references. By default, store weakrefs to avoid OOM.
                try:
                    if getattr(self, 'keep_activations_strong', False):
                        # strong refs with optional size cutoff
                        saved = []
                        total_saved = sum((t.numel() * t.element_size()) for t in self.saved_activation_refs.values() if isinstance(t, torch.Tensor)) if self.saved_activation_refs else 0
                        if isinstance(output, (list, tuple)):
                            for o in output:
                                if not isinstance(o, torch.Tensor):
                                    continue
                                o_bytes = int(o.numel() * o.element_size())
                                if self.max_saved_activation_bytes is not None and (total_saved + o_bytes) > self.max_saved_activation_bytes:
                                    # skip storing further tensors
                                    continue
                                saved.append(o)
                                total_saved += o_bytes
                        else:
                            o = output
                            o_bytes = int(o.numel() * o.element_size())
                            if self.max_saved_activation_bytes is None or (total_saved + o_bytes) <= self.max_saved_activation_bytes:
                                saved = o
                        self.saved_activation_refs[m_key] = saved
                    else:
                        import weakref
                        if isinstance(output, (list, tuple)):
                            self.saved_activation_refs[m_key] = [weakref.ref(o) for o in output if isinstance(o, torch.Tensor)]
                        else:
                            self.saved_activation_refs[m_key] = weakref.ref(output)
                except Exception:
                    self.saved_activation_refs[m_key] = None

            # Receptive field bookkeeping: compute per-spatial-dim (h,w)
            # Attempt to derive input receptive field from input tensors
            in_rfs = []
            if isinstance(input, tuple):
                for inp in input:
                    if isinstance(inp, torch.Tensor):
                        rf_info = self.tensor_rf_map.get(id(inp))
                        if rf_info is not None:
                            in_rfs.append(rf_info)
            elif isinstance(input, torch.Tensor):
                rf_info = self.tensor_rf_map.get(id(input))
                if rf_info is not None:
                    in_rfs.append(rf_info)

            # default input rf if none
            if not in_rfs:
                # spatial receptive field (h,w), jump (h,w), start (h,w)
                in_rf = {'rf': (1.0, 1.0), 'jump': (1.0, 1.0), 'start': (0.5, 0.5)}
            else:
                # for multiple inputs (branches), take the maximal rf/jump alignment
                # We'll take the max rf and max jump, and average starts
                rfs = [r['rf'] for r in in_rfs]
                jumps = [r['jump'] for r in in_rfs]
                starts = [r['start'] for r in in_rfs]

                # If all jumps and starts align within small tolerance, prefer that
                def all_close(vals, tol=1e-6):
                    first = vals[0]
                    return all(abs(v - first) < tol for v in vals)

                jumps_h = [t[0] for t in jumps]
                jumps_w = [t[1] for t in jumps]
                starts_h = [t[0] for t in starts]
                starts_w = [t[1] for t in starts]

                if all_close(jumps_h) and all_close(jumps_w) and all_close(starts_h) and all_close(starts_w):
                    # take max RF but keep common jump/start
                    in_rf = {
                        'rf': (max([t[0] for t in rfs]), max([t[1] for t in rfs])),
                        'jump': (jumps_h[0], jumps_w[0]),
                        'start': (starts_h[0], starts_w[0])
                    }
                else:
                    # mismatched receptive fields across branches — fall back to conservative merge and warn
                    in_rf = {
                        'rf': (max([t[0] for t in rfs]), max([t[1] for t in rfs])),
                        'jump': (max([t[0] for t in jumps]), max([t[1] for t in jumps])),
                        'start': (sum([t[0] for t in starts]) / len(starts), sum([t[1] for t in starts]) / len(starts))
                    }
                    try:
                        self.init_warnings[m_key] = 'mismatched receptive field across input branches'
                    except Exception:
                        pass
            
            self.summary_data[m_key] = {
                "layer_name": class_name,
                "input_shape": [list(i.size()) if isinstance(i, torch.Tensor) else str(i) for i in input] if isinstance(input, tuple) else list(input.size()) if isinstance(input, torch.Tensor) else str(input),
                "output_shape": output_shape,
                "params": params,
                "trainable": trainable,
                "output_size_mb": output_size / (1024 ** 2),
                "activation_bytes": act_bytes,
                "module": module
            }
            
            self.total_params += params
            self.trainable_params += trainable
            self.total_output_size += output_size

            # compute receptive field for this layer if conv/pool
            try:
                # handle 2D conv/pool
                if isinstance(module, nn.Conv2d):
                    k = module.kernel_size
                    s = module.stride
                    p = module.padding
                    d = module.dilation

                    # support tuple or int
                    kh, kw = (k if isinstance(k, tuple) else (k, k))
                    sh, sw = (s if isinstance(s, tuple) else (s, s))
                    ph, pw = (p if isinstance(p, tuple) else (p, p))
                    dh, dw = (d if isinstance(d, tuple) else (d, d))

                    in_rf_h, in_rf_w = in_rf['rf']
                    in_jump_h, in_jump_w = in_rf['jump']
                    in_start_h, in_start_w = in_rf['start']

                    k_eff_h = kh + (kh - 1) * (dh - 1)
                    k_eff_w = kw + (kw - 1) * (dw - 1)

                    out_rf_h = in_rf_h + (k_eff_h - 1) * in_jump_h
                    out_rf_w = in_rf_w + (k_eff_w - 1) * in_jump_w

                    out_jump_h = in_jump_h * sh
                    out_jump_w = in_jump_w * sw

                    out_start_h = in_start_h + ((k_eff_h - 1) / 2.0 - ph) * in_jump_h
                    out_start_w = in_start_w + ((k_eff_w - 1) / 2.0 - pw) * in_jump_w

                    self.receptive_field[m_key] = {
                        'rf': (float(out_rf_h), float(out_rf_w)),
                        'jump': (float(out_jump_h), float(out_jump_w)),
                        'start': (float(out_start_h), float(out_start_w))
                    }

                    # map output tensor id(s) to rf info
                    if isinstance(output, torch.Tensor):
                        self.tensor_rf_map[id(output)] = self.receptive_field[m_key]
                    else:
                        # list/tuple outputs
                        try:
                            for o in output:
                                if isinstance(o, torch.Tensor):
                                    self.tensor_rf_map[id(o)] = self.receptive_field[m_key]
                        except Exception:
                            pass

                elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                    k = module.kernel_size
                    s = module.stride if module.stride is not None else module.kernel_size
                    p = module.padding if hasattr(module, 'padding') else 0

                    kh, kw = (k if isinstance(k, tuple) else (k, k))
                    sh, sw = (s if isinstance(s, tuple) else (s, s))
                    ph, pw = (p if isinstance(p, tuple) else (p, p))

                    in_rf_h, in_rf_w = in_rf['rf']
                    in_jump_h, in_jump_w = in_rf['jump']
                    in_start_h, in_start_w = in_rf['start']

                    k_eff_h = kh
                    k_eff_w = kw

                    out_rf_h = in_rf_h + (k_eff_h - 1) * in_jump_h
                    out_rf_w = in_rf_w + (k_eff_w - 1) * in_jump_w

                    out_jump_h = in_jump_h * sh
                    out_jump_w = in_jump_w * sw

                    out_start_h = in_start_h + ((k_eff_h - 1) / 2.0 - ph) * in_jump_h
                    out_start_w = in_start_w + ((k_eff_w - 1) / 2.0 - pw) * in_jump_w

                    self.receptive_field[m_key] = {
                        'rf': (float(out_rf_h), float(out_rf_w)),
                        'jump': (float(out_jump_h), float(out_jump_w)),
                        'start': (float(out_start_h), float(out_start_w))
                    }

                    if isinstance(output, torch.Tensor):
                        self.tensor_rf_map[id(output)] = self.receptive_field[m_key]
            except Exception:
                # fallback: map outputs to input rf
                try:
                    if isinstance(output, torch.Tensor):
                        self.tensor_rf_map[id(output)] = in_rf
                except Exception:
                    pass
        
        hooks = []
        for name, layer in self.model.named_modules():
            # Skip container modules
            if list(layer.children()):
                continue
            hooks.append(layer.register_forward_hook(hook_fn))
        
        return hooks

    def _compute_receptive_field(self):
        """
        Compute receptive field, effective stride (jump), and start offset for conv/pool layers.

        Uses standard receptive field bookkeeping:
          k_eff = k + (k-1)*(d-1)
          new_rf = rf + (k_eff - 1) * jump
          new_jump = jump * stride
          new_start = start + ((k_eff - 1)/2 - padding) * jump
        """
        rf = 1.0
        jump = 1.0
        start = 0.5

        for key, info in self.summary_data.items():
            mod = info.get("module")
            if mod is None:
                self.receptive_field[key] = {"rf": rf, "jump": jump, "start": start}
                continue

            # default values
            k = 1
            s = 1
            p = 0
            d = 1

            if isinstance(mod, nn.Conv2d):
                k = mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else mod.kernel_size
                s = mod.stride[0] if isinstance(mod.stride, tuple) else mod.stride
                p = mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding
                d = mod.dilation[0] if isinstance(mod.dilation, tuple) else mod.dilation
            elif isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d)):
                k = mod.kernel_size if isinstance(mod.kernel_size, int) else (mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else 1)
                s = mod.stride if mod.stride is not None else k
                s = s if isinstance(s, int) else s[0]
                p = mod.padding if hasattr(mod, 'padding') else 0
                p = p if isinstance(p, int) else p[0]

            k_eff = k + (k - 1) * (d - 1)
            new_rf = rf + (k_eff - 1) * jump
            new_jump = jump * s
            new_start = start + ((k_eff - 1) / 2.0 - p) * jump

            # store values for this layer
            self.receptive_field[key] = {"rf": float(new_rf), "jump": float(new_jump), "start": float(new_start)}

            # update running
            rf, jump, start = new_rf, new_jump, new_start

    def _dryrun_initialization_and_gradients(self, warn_zero_grad: bool = True, warn_large_grad: bool = True):
        """
        Perform a dry-run forward+backward to inspect initial gradients and weight scales.

        Flags layers with zero gradients, extremely large gradients, or suspicious init scaling
        (compared to Xavier/He heuristics).
        """
        # Only run if we have input size
        if self.input_size is None:
            return

        # Create small random input
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        self.model.to(self.device)

        # Ensure train mode so autograd keeps activations
        self.model.train()

        # Zero grads
        self.model.zero_grad(set_to_none=True)

        # Forward
        output = self.model(x)
        loss = output.sum() if not isinstance(output, (list, tuple)) else output[0].sum()

        # Backward
        loss.backward()

        # Analyze per-layer gradients and initial weight scales
        for key, info in self.summary_data.items():
            module = info.get('module')
            if module is None:
                continue

            msg_parts = []
            # check parameter gradients
            grads_found = False
            large_grad = False
            zero_grad = True
            for p in module.parameters():
                if p.grad is None:
                    continue
                grads_found = True
                g = p.grad.detach()
                g_abs_max = float(g.abs().max().item()) if g.numel() > 0 else 0.0
                g_mean = float(g.mean().item()) if g.numel() > 0 else 0.0
                if g_abs_max > self.grad_large_threshold and warn_large_grad:
                    large_grad = True
                if g_abs_max > self.grad_zero_tol:
                    zero_grad = False

            if not grads_found:
                msg_parts.append('no parameter gradients')
            else:
                if zero_grad and warn_zero_grad:
                    msg_parts.append('zero gradients')
                if large_grad:
                    msg_parts.append('very large gradients')

            # initialization scale check (heuristic)
            try:
                for p in module.parameters():
                    w = p.detach()
                    if w.numel() == 0:
                        continue
                    std = float(w.std().item())
                    # compute fan_in, fan_out for Linear/Conv
                    fan_in = None
                    fan_out = None
                    if isinstance(module, (nn.Conv2d,)):
                        k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                        cin = module.in_channels
                        cout = module.out_channels
                        fan_in = cin * k * k
                        fan_out = cout * k * k
                    elif isinstance(module, (nn.Linear,)):
                        fan_in = w.size(1) if w.dim() >= 2 else w.numel()
                        fan_out = w.size(0) if w.dim() >= 2 else w.numel()

                    if fan_in is not None and fan_out is not None:
                        # Xavier std and He std
                        xavier_std = (2.0 / (fan_in + fan_out)) ** 0.5
                        he_std = (2.0 / fan_in) ** 0.5
                        # if std is orders of magnitude off, warn
                        if std > max(xavier_std, he_std) * self.init_std_warn_multiply:
                            msg_parts.append(f'large init std ({std:.3e})')
                        elif std < min(xavier_std, he_std) * self.init_std_warn_min_mult:
                            msg_parts.append(f'small init std ({std:.3e})')
                    break
            except Exception:
                pass

            if msg_parts:
                self.init_warnings[key] = '; '.join(msg_parts)

        # clear gradients to avoid side effects
        self.model.zero_grad(set_to_none=True)
    
    def _register_gradient_hooks(self):
        """Register hooks to track gradient statistics"""
        def grad_hook_fn(module, grad_input, grad_output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            
            # Calculate gradient variance
            if grad_output[0] is not None:
                grad_var = grad_output[0].var().item()
                grad_mean = grad_output[0].mean().item()
                grad_max = grad_output[0].abs().max().item()
                
                # Find corresponding key in summary_data
                for key in self.summary_data.keys():
                    if key.startswith(class_name):
                        self.gradient_stats[key] = {
                            "grad_variance": grad_var,
                            "grad_mean": grad_mean,
                            "grad_max": grad_max
                        }
                        break
        
        hooks = []
        for layer in self.model.modules():
            if list(layer.children()):
                continue
            hooks.append(layer.register_full_backward_hook(grad_hook_fn))
        
        return hooks
    
    def _analyze_model(self):
        """Run model analysis"""
        if self.input_size is None:
            # Can't do shape inference without input size
            self._analyze_without_forward()
            return
        
        # Register hooks
        hooks = self._register_hooks([])
        
        # Create dummy input
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        self.model.to(self.device)
        
        # Forward pass
        if self.track_gradients:
            grad_hooks = self._register_gradient_hooks()
            self.model.train()
            output = self.model(x)
            
            # Backward pass to get gradients
            if isinstance(output, (list, tuple)):
                loss = output[0].sum()
            else:
                loss = output.sum()
            loss.backward()
            
            # Remove gradient hooks
            for h in grad_hooks:
                h.remove()
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(x)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        # Compute receptive field metadata
        try:
            # already computed during forward hooks; ensure mapping exists
            # fallback to separate computation if needed
            if not self.receptive_field:
                self._compute_receptive_field()
        except Exception:
            pass

        # Memory profile estimation (approximate)
        try:
            param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
            activation_bytes = 0
            for k, info in self.summary_data.items():
                activation_bytes += int(info.get('activation_bytes', 0))

            grad_bytes = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_bytes += p.grad.numel() * p.grad.element_size()

            self.memory_profile = {
                'parameter_bytes': int(param_bytes),
                'activation_bytes_estimate': int(activation_bytes),
                'gradient_bytes': int(grad_bytes),
                'total_estimated_bytes': int(param_bytes + activation_bytes + grad_bytes),
                'saved_activations': len(self.saved_activation_refs) if hasattr(self, 'saved_activation_refs') else 0
            }
        except Exception:
            self.memory_profile = {}
    
    def _analyze_without_forward(self):
        """Analyze model structure without forward pass"""
        for name, module in self.model.named_modules():
            if list(module.children()):
                continue
            
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            self.summary_data[name] = {
                "layer_name": class_name,
                "input_shape": "N/A",
                "output_shape": "N/A",
                "params": params,
                "trainable": trainable,
                "output_size_mb": 0.0,
                "module": module
            }
            
            self.total_params += params
            self.trainable_params += trainable
    
    def get_bottlenecks(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Identify bottleneck layers based on multiple criteria.
        
        A bottleneck is defined as a layer with:
        1. High parameter count (>10% of total)
        2. High gradient variance (if tracking enabled)
        3. Large output size
        
        Args:
            top_n: Number of top bottlenecks to return
        
        Returns:
            List of dictionaries containing bottleneck information
        """
        bottlenecks = []
        
        for key, info in self.summary_data.items():
            score = 0.0
            reasons = []
            
            # Parameter count criterion
            if self.total_params > 0:
                param_ratio = info["params"] / self.total_params
                if param_ratio > self.param_ratio_bottleneck:  # configurable
                    score += param_ratio * 100
                    reasons.append(f"High params ({param_ratio*100:.1f}%)")
            
            # Gradient variance criterion (if available)
            if key in self.gradient_stats:
                grad_var = self.gradient_stats[key]["grad_variance"]
                if grad_var > 1.0:  # High variance
                    score += np.log10(grad_var + 1) * 10
                    reasons.append(f"High grad variance ({grad_var:.2e})")
            
            # Output size criterion
            if info.get("output_size_mb", 0) > self.activation_bottleneck_mb:  # configurable
                score += info.get("output_size_mb", 0)
                reasons.append(f"Large output ({info.get('output_size_mb',0):.1f}MB)")
            
            if score > 0:
                bottlenecks.append({
                    "layer": key,
                    "layer_name": info["layer_name"],
                    "score": score,
                    "reasons": reasons,
                    "params": info["params"],
                    "output_shape": info["output_shape"]
                })
        
        # Sort by score and return top N
        bottlenecks.sort(key=lambda x: x["score"], reverse=True)
        return bottlenecks[:top_n]
    
    def show(self, show_bottlenecks: bool = True):
        """
        Display the model summary in a pretty-printed format.
        
        Args:
            show_bottlenecks: Whether to show bottleneck analysis
        """
        print("=" * 100)
        print(f"{'Model Summary':^100}")
        print("=" * 100)
        
        # Header
        print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':<15} {'Trainable':<10}")
        print("-" * 100)
        
        # Layers
        for key, info in self.summary_data.items():
            layer_name = f"{info['layer_name']}"
            output_shape = str(info['output_shape'])
            if len(output_shape) > 24:
                output_shape = output_shape[:21] + "..."
            
            params = f"{info['params']:,}"
            trainable = "Yes" if info['trainable'] > 0 else "No"
            
            print(f"{layer_name:<30} {output_shape:<25} {params:<15} {trainable:<10}")
            
            # Show gradient stats if available
            if key in self.gradient_stats:
                grad_info = self.gradient_stats[key]
                print(f"  └─ Gradient: var={grad_info['grad_variance']:.2e}, "
                      f"mean={grad_info['grad_mean']:.2e}, max={grad_info['grad_max']:.2e}")
        
        print("=" * 100)
        print(f"Total params: {self.total_params:,}")
        print(f"Trainable params: {self.trainable_params:,}")
        print(f"Non-trainable params: {self.total_params - self.trainable_params:,}")
        print(f"Total output size: {self.total_output_size / (1024**2):.2f} MB")
        print("=" * 100)
        
        # Show bottlenecks
        if show_bottlenecks:
            bottlenecks = self.get_bottlenecks()
            if bottlenecks:
                print(f"\n{'[!] Detected Bottleneck Layers':^100}")
                print("=" * 100)
                
                for i, bn in enumerate(bottlenecks, 1):
                    print(f"\n{i}. {bn['layer']} ({bn['layer_name']})")
                    print(f"   Score: {bn['score']:.2f}")
                    print(f"   Reasons: {', '.join(bn['reasons'])}")
                    print(f"   Parameters: {bn['params']:,}")
                    print(f"   Output Shape: {bn['output_shape']}")
                
                print("\n" + "=" * 100)
        # Show initialization warnings (dry-run)
        if self.init_warnings:
            print(f"\n{'[!] Initialization / Gradient Warnings':^100}")
            print("=" * 100)
            for k, v in self.init_warnings.items():
                print(f" - {k}: {v}")
            print("\n" + "=" * 100)

        # Show receptive field summary
        if self.receptive_field:
            print(f"\n{'Receptive Field (rf/jump/start)':^100}")
            print("=" * 100)
            def _fmt_val(x):
                try:
                    if isinstance(x, (tuple, list)):
                        return '(' + ', '.join(f"{float(t):.2f}" for t in x) + ')'
                    else:
                        return f"{float(x):.2f}"
                except Exception:
                    return str(x)

            for k, v in self.receptive_field.items():
                rf = _fmt_val(v.get('rf'))
                jump = _fmt_val(v.get('jump'))
                start = _fmt_val(v.get('start'))
                print(f" - {k}: rf={rf}, jump={jump}, start={start}")
            print("\n" + "=" * 100)

        # Memory profile
        if self.memory_profile:
            print(f"\n{'Memory Profile (bytes)':^100}")
            print("=" * 100)
            for kk, vv in self.memory_profile.items():
                print(f" - {kk}: {vv}")
            print("\n" + "=" * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export summary as a dictionary.
        
        Returns:
            Dictionary containing all summary information
        """
        return {
            "layers": self.summary_data,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "total_output_size_mb": self.total_output_size / (1024**2),
            "gradient_stats": self.gradient_stats,
            "bottlenecks": self.get_bottlenecks(),
            "init_warnings": self.init_warnings,
            "receptive_field": self.receptive_field,
            "memory_profile": self.memory_profile
        }
    
    def save_to_file(self, filename: str = "model_summary.txt"):
        """
        Save the summary to a text file.
        
        Args:
            filename: Output file path
        """
        import sys
        from io import StringIO
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        self.show()
        
        sys.stdout = old_stdout
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(captured_output.getvalue())
        
        print(f"Summary saved to {filename}")
    
    def analyze_loss_curve(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        window_size: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze the nature of training loss curve(s) to understand training dynamics.
        
        This method identifies patterns such as:
        - Convergence: Loss is decreasing and stabilizing
        - Divergence: Loss is increasing (training instability)
        - Oscillation: High variance in loss values
        - Plateau: Loss has stopped improving
        - Overfitting: Training loss decreases while validation loss increases
        
        Args:
            train_losses: List of training loss values over epochs
            val_losses: Optional list of validation loss values over epochs
            window_size: Window size for moving average smoothing (default: 5)
            verbose: Whether to print analysis results (default: True)
        
        Returns:
            Dictionary containing:
            - 'status': Overall training status
            - 'trend': Direction of loss movement
            - 'stability': Measure of oscillation/variance
            - 'recommendations': List of suggestions
            - 'metrics': Detailed metrics about the loss curve
        
        Example:
            >>> train_losses = [2.5, 2.1, 1.8, 1.6, 1.5, 1.45, 1.42, 1.41]
            >>> val_losses = [2.6, 2.2, 1.9, 1.7, 1.8, 1.9, 2.0, 2.1]
            >>> result = smart_summary.analyze_loss_curve(train_losses, val_losses)
            >>> print(result['status'])  # 'Overfitting Detected'
        """
        if not train_losses or len(train_losses) < 2:
            return {
                "status": "Insufficient Data",
                "trend": "Unknown",
                "stability": "Unknown",
                "recommendations": ["Provide at least 2 epochs of training data"],
                "metrics": {}
            }
        
        train_losses = np.array(train_losses, dtype=np.float32)
        n_epochs = len(train_losses)
        
        # Calculate moving average for smoothing
        def moving_average(data, window):
            if len(data) < window:
                return data
            weights = np.ones(window) / window
            return np.convolve(data, weights, mode='valid')
        
        smoothed_train = moving_average(train_losses, min(window_size, len(train_losses)))
        
        # Calculate key metrics
        initial_loss = float(train_losses[0])
        final_loss = float(train_losses[-1])
        min_loss = float(np.min(train_losses))
        max_loss = float(np.max(train_losses))
        
        # Calculate trend (slope)
        epochs_idx = np.arange(len(smoothed_train))
        if len(smoothed_train) > 1:
            trend_slope = np.polyfit(epochs_idx, smoothed_train, 1)[0]
        else:
            trend_slope = 0.0
        
        # Calculate variance and coefficient of variation
        loss_variance = float(np.var(train_losses))
        loss_std = float(np.std(train_losses))
        mean_loss = float(np.mean(train_losses))
        coef_variation = (loss_std / mean_loss * 100) if mean_loss != 0 else 0.0
        
        # Calculate recent trend (last 30% of epochs)
        recent_window = max(3, int(n_epochs * 0.3))
        recent_losses = train_losses[-recent_window:]
        recent_slope = np.polyfit(np.arange(len(recent_losses)), recent_losses, 1)[0]
        
        # Improvement percentage
        improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss != 0 else 0.0
        
        # Determine training status
        status_messages = []
        recommendations = []
        
        # Check for divergence
        if trend_slope > 0.001 or improvement < -5:
            status = "Diverging"
            status_messages.append("⚠️ Loss is INCREASING - training instability detected")
            recommendations.extend([
                "Reduce learning rate significantly (try 0.1x current value)",
                "Check for gradient explosion (use gradient clipping)",
                "Verify data preprocessing and labels are correct",
                "Consider using a simpler model architecture"
            ])
        
        # Check for plateau
        elif abs(recent_slope) < 0.0001 and n_epochs > 10:
            status = "Plateau"
            status_messages.append("📊 Loss has plateaued - minimal improvement in recent epochs")
            recommendations.extend([
                "Consider reducing learning rate to escape plateau",
                "Try learning rate scheduling or cyclical learning rates",
                "Check if model has sufficient capacity",
                "Evaluate if current performance is acceptable"
            ])
        
        # Check for high oscillation
        elif coef_variation > 20:
            status = "Oscillating"
            status_messages.append("📈 High oscillation in loss values detected")
            recommendations.extend([
                "Reduce learning rate to stabilize training",
                "Increase batch size for more stable gradients",
                "Consider using gradient clipping",
                "Check for outliers or noise in training data"
            ])
        
        # Check for good convergence
        elif improvement > 20 and abs(recent_slope) < 0.01:
            status = "Converging Well"
            status_messages.append("✅ Loss is converging nicely")
            recommendations.extend([
                "Training appears healthy - continue monitoring",
                "Consider early stopping if validation loss plateaus"
            ])
        
        # Slow improvement
        elif improvement > 5:
            status = "Converging Slowly"
            status_messages.append("🐌 Loss is decreasing but slowly")
            recommendations.extend([
                "Consider slightly increasing learning rate",
                "Verify model has sufficient capacity for the task",
                "Check if data preprocessing is optimal"
            ])
        
        else:
            status = "Stable"
            status_messages.append("📍 Loss is relatively stable")
            recommendations.append("Continue monitoring training progress")
        
        # Analyze validation losses if provided
        overfitting_detected = False
        if val_losses is not None and len(val_losses) >= 2:
            val_losses = np.array(val_losses, dtype=np.float32)
            
            if len(val_losses) != len(train_losses):
                recommendations.append(
                    f"⚠️ Warning: Train and validation losses have different lengths "
                    f"({len(train_losses)} vs {len(val_losses)})"
                )
            
            val_initial = float(val_losses[0])
            val_final = float(val_losses[-1])
            val_min = float(np.min(val_losses))
            
            # Check for overfitting: train loss decreasing, val loss increasing
            recent_val = val_losses[-recent_window:]
            val_recent_slope = np.polyfit(np.arange(len(recent_val)), recent_val, 1)[0]
            
            if recent_slope < -0.001 and val_recent_slope > 0.005:
                status = "Overfitting Detected"
                overfitting_detected = True
                status_messages.append(
                    "⚠️ OVERFITTING: Training loss decreasing but validation loss increasing"
                )
                recommendations.extend([
                    "Stop training or use early stopping",
                    "Increase regularization (L1/L2, dropout)",
                    "Add more training data if possible",
                    "Reduce model complexity",
                    "Use data augmentation"
                ])
            
            # Check for validation loss not improving
            elif val_final > val_min * 1.05 and n_epochs > 5:
                status_messages.append(
                    "⚠️ Validation loss stopped improving (potential overfitting)"
                )
                if not overfitting_detected:
                    recommendations.extend([
                        "Consider early stopping",
                        "Monitor validation loss closely"
                    ])
        
        # Determine trend description
        if trend_slope < -0.01:
            trend = "Decreasing"
        elif trend_slope > 0.01:
            trend = "Increasing"
        else:
            trend = "Stable"
        
        # Determine stability
        if coef_variation < 5:
            stability = "Very Stable"
        elif coef_variation < 15:
            stability = "Stable"
        elif coef_variation < 30:
            stability = "Moderately Unstable"
        else:
            stability = "Highly Unstable"
        
        # Compile metrics
        metrics = {
            "n_epochs": n_epochs,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "min_loss": min_loss,
            "max_loss": max_loss,
            "improvement_percent": improvement,
            "trend_slope": float(trend_slope),
            "recent_slope": float(recent_slope),
            "variance": loss_variance,
            "std_dev": loss_std,
            "mean_loss": mean_loss,
            "coefficient_of_variation": coef_variation
        }
        
        if val_losses is not None:
            metrics["val_initial_loss"] = float(val_losses[0])
            metrics["val_final_loss"] = float(val_losses[-1])
            metrics["val_min_loss"] = float(np.min(val_losses))
            metrics["val_improvement_percent"] = (
                (float(val_losses[0]) - float(val_losses[-1])) / float(val_losses[0]) * 100
                if val_losses[0] != 0 else 0.0
            )
        
        result = {
            "status": status,
            "trend": trend,
            "stability": stability,
            "recommendations": recommendations,
            "metrics": metrics
        }
        
        # Print analysis if verbose
        if verbose:
            print("\n" + "="*80)
            print(f"{'Loss Curve Analysis':^80}")
            print("="*80)
            
            for msg in status_messages:
                print(f"\n{msg}")
            
            print(f"\n📊 Training Statistics:")
            print(f"   • Epochs: {n_epochs}")
            print(f"   • Initial Loss: {initial_loss:.4f}")
            print(f"   • Final Loss: {final_loss:.4f}")
            print(f"   • Minimum Loss: {min_loss:.4f}")
            print(f"   • Improvement: {improvement:.2f}%")
            print(f"   • Trend: {trend} (slope: {trend_slope:.6f})")
            print(f"   • Stability: {stability} (CV: {coef_variation:.2f}%)")
            
            if val_losses is not None:
                print(f"\n📊 Validation Statistics:")
                print(f"   • Initial Val Loss: {metrics['val_initial_loss']:.4f}")
                print(f"   • Final Val Loss: {metrics['val_final_loss']:.4f}")
                print(f"   • Minimum Val Loss: {metrics['val_min_loss']:.4f}")
                print(f"   • Val Improvement: {metrics['val_improvement_percent']:.2f}%")
            
            if recommendations:
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            print("\n" + "="*80)
        
        return result
