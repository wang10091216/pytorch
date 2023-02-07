from typing import Any, Dict, Optional
from torch.nn.utils.parametrize import type_before_parametrizations, is_parametrized
from itertools import chain

from torch import nn

__all__ = [
    "module_contains_param",
    "swap_module",
    "module_to_fqn",
    "fqn_to_module",
    "get_arg_info_from_tensor_fqn",
    "FakeSparsity",
]


def module_contains_param(module, parametrization):
    if is_parametrized(module):
        # see if any of the module tensors have a parametriztion attached that matches the one passed in
        return any(
            [
                any(isinstance(param, parametrization) for param in param_list)
                for key, param_list in module.parametrizations.items()
            ]
        )
    return False

def swap_module(mod, mapping, custom_module_class_mapping):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.
    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module
    Return:
        The corresponding quantized module of `mod`
    """
    new_mod = mod
    swapped = False
    if type_before_parametrizations(mod) in custom_module_class_mapping:
        new_mod = custom_module_class_mapping[type_before_parametrizations(mod)].from_dense(mod)
        swapped = True
    elif type_before_parametrizations(mod) in mapping:
        qmod = mapping[type_before_parametrizations(mod)]
        new_mod = qmod.from_dense(mod)
        swapped = True

    if swapped:
        # Preserve module's pre forward hooks. They'll be called on quantized input
        for pre_hook_fn in mod._forward_pre_hooks.values():
            new_mod.register_forward_pre_hook(pre_hook_fn)
        # Preserve module's post forward hooks except _observer_forward_hook
        # After convert they'll work with quantized output
        for hook_fn in mod._forward_hooks.values():
            if hook_fn is not _observer_forward_hook:
                new_mod.register_forward_hook(hook_fn)

        # respect device affinity when swapping modules
        devices = {p.device for p in chain(mod.parameters(), mod.buffers())}
        assert len(devices) <= 1, (
            "swap_module only works with cpu or single-device CUDA modules, "
            "but got devices {}".format(devices)
        )
        device = next(iter(devices)) if len(devices) > 0 else None
        if device:
            new_mod.to(device)
    return new_mod

def module_to_fqn(model: nn.Module, module: nn.Module, prefix: str = "") -> Optional[str]:
    """
    Returns the fqn for a module or None if module not a descendent of model.
    """
    if module is model:
        return ""
    for name, child in model.named_children():
        fqn = module_to_fqn(child, module, ".")
        if isinstance(fqn, str):
            return prefix + name + fqn
    return None


def fqn_to_module(model: Optional[nn.Module], path: str) -> Optional[nn.Module]:
    """
    Given an fqn, returns the corresponding module or tensor or None if the fqn given by `path`
    doesn't correspond to anything. Similar to model.get_submodule(path) but works for tensors.
    """
    if path != "":
        for name in path.split("."):
            model = getattr(model, name, None)
    return model


def get_arg_info_from_tensor_fqn(model: nn.Module, tensor_fqn: str) -> Dict[str, Any]:
    """
    Uses tensor_fqn to obtain a dict containing module_fqn, module and tensor_name
    """
    # string manip to split tensor_fqn into module_fqn and tensor_name
    # if tensor_fqn is 'weight' then module_fqn and tensor_name are '' and 'weight'
    # if tensor_fqn is 'linear.weight' then module_fqn and tensor_name are 'linear' and 'weight'
    tensor_name = tensor_fqn.split(".")[-1]
    module_fqn = tensor_fqn[: -len(tensor_name) - ("." in tensor_fqn)]

    module = fqn_to_module(model, module_fqn)

    return {
        "module_fqn": module_fqn,
        "module": module,
        "tensor_name": tensor_name,
        "tensor_fqn": tensor_fqn,
    }


# Parametrizations
class FakeSparsity(nn.Module):
    r"""Parametrization for the weights. Should be attached to the 'weight' or
    any other parmeter that requires a mask applied to it.

    Note::

        Once the mask is passed, the variable should not change the id. The
        contents of the mask can change, but the mask reference itself should
        not.
    """
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        assert self.mask.shape == x.shape
        return self.mask * x

    def state_dict(self, *args, **kwargs):
        # We don't want to let the parametrizations to save the mask.
        # That way we make sure that the linear module doesn't store the masks
        # alongside their parametrizations.
        return {}
