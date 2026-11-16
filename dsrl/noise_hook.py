"""Inject a chosen initial noise into a flow-matching action head without editing it.

Both VLAs draw the initial latent with a single torch call right before the
Euler loop:

  * Xiaomi-Robotics-1 (modeling_mibot.py):  x = torch.randn_like(action_mask)
  * RLDX-1 (rldx/model/core/rldx.py):       actions = torch.randn(size=(B, H, D), ...)

DSRL only needs to choose that latent, so we temporarily replace the torch
function and hand back our tensor for the first call whose shape matches.
`hits` lets the caller verify that the injection really happened (a silent
miss would mean we are evaluating the unsteered policy).
"""

from __future__ import annotations

import contextlib

import torch


def _randn_size(args, kwargs):
    if "size" in kwargs:
        return tuple(kwargs["size"])
    if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
        return tuple(args[0])
    return tuple(args)


@contextlib.contextmanager
def inject_initial_noise(noise: torch.Tensor | None, fn_name: str):
    """fn_name: 'randn' or 'randn_like'. Yields a dict with a 'hits' counter."""
    state = {"hits": 0}
    if noise is None:
        yield state
        return

    shape = tuple(noise.shape)
    original = getattr(torch, fn_name)

    if fn_name == "randn":

        def patched(*args, **kwargs):
            if state["hits"] == 0 and _randn_size(args, kwargs) == shape:
                state["hits"] += 1
                return noise.to(device=kwargs.get("device"), dtype=kwargs.get("dtype")).clone()
            return original(*args, **kwargs)

    elif fn_name == "randn_like":

        def patched(input, *args, **kwargs):
            if state["hits"] == 0 and tuple(input.shape) == shape:
                state["hits"] += 1
                return noise.to(
                    device=kwargs.get("device", input.device),
                    dtype=kwargs.get("dtype", input.dtype),
                ).clone()
            return original(input, *args, **kwargs)

    else:
        raise ValueError(fn_name)

    setattr(torch, fn_name, patched)
    try:
        yield state
    finally:
        setattr(torch, fn_name, original)


def require_single_hit(state: dict, noise, what: str) -> None:
    if noise is not None and state["hits"] != 1:
        raise RuntimeError(
            f"{what}: initial-noise injection did not fire (hits={state['hits']}). "
            "The model's sampling code no longer matches the hook; steering would be a no-op."
        )
