from typing import Callable

import torch

F_DEBUG_PRINT = False


def checkpoint(fn: Callable, *args, **kwargs):
    # If gradient tracking is enabled and any input requires grad
    grad_enabled = torch.is_grad_enabled()
    input_requires_grad = any(isinstance(a, torch.Tensor) and a.requires_grad for a in args)

    if grad_enabled and input_requires_grad:
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)
    elif input_requires_grad:
        # Defensive programming: log a warning if checkpointing is skipped due to grad being disabled
        if F_DEBUG_PRINT:
            print("[DEBUG] Skipping checkpoint because grad mode is off but input requires grad.")
    return fn(*args)
