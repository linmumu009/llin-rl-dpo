"""Optional runtime patches for local Ascend/ms-swift experiments.

This module is loaded automatically by Python when its directory is on
PYTHONPATH. Keep patches behind explicit environment flags so normal runs stay
as close to upstream behavior as possible.
"""

from __future__ import annotations

import os


def _patch_swift_model_load_state_dict_assign() -> None:
    import swift.tuners.base as swift_base

    SwiftModel = swift_base.SwiftModel
    if getattr(SwiftModel.load_state_dict, "_llin_accepts_assign", False):
        return

    original = SwiftModel.load_state_dict

    def load_state_dict_compat(self, state_dict, strict=True, adapter_name=None, *args, **kwargs):
        assign = kwargs.pop("assign", False)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected load_state_dict keyword(s): {unexpected}")

        if adapter_name is not None:
            return original(self, state_dict, strict=strict, adapter_name=adapter_name)

        try:
            incompatible_keys = self.base_model.load_state_dict(state_dict, False, assign=assign)
        except TypeError:
            incompatible_keys = self.base_model.load_state_dict(state_dict, False)

        if incompatible_keys and len(incompatible_keys[1]) > 0:
            swift_base.logger.error(f"Load state dict with unexpected keys: {incompatible_keys[1]}")
        return incompatible_keys

    load_state_dict_compat._llin_accepts_assign = True
    SwiftModel.load_state_dict = load_state_dict_compat
    print("[llin-rl-dpo] Patched SwiftModel.load_state_dict to accept assign=...", flush=True)


if os.environ.get("LLIN_SWIFTMODEL_ASSIGN_PATCH") == "1":
    _patch_swift_model_load_state_dict_assign()
