from contextlib import contextmanager
import re
import threading
from typing import Iterable

import torch


_ALIAS_LOCK = threading.RLock()


def _supports_named_safe_globals() -> bool:
    """Return whether PyTorch accepts ``(callable, full_path)`` entries."""

    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None:
        raise RuntimeError(f"Cannot parse PyTorch version {torch.__version__!r}.")
    return tuple(int(value) for value in match.groups()) >= (2, 6)


@contextmanager
def aliased_safe_globals(
    classes: Iterable[type], legacy_module: str
):
    """Allow current classes under their pre-refactor pickle paths.

    PyTorch 2.6 added named safe-global aliases. TMol supports PyTorch 2.5,
    whose weights-only unpickler accepts callables but not alias tuples. On
    2.5, temporarily expose each class under the trusted legacy module name.
    """

    trusted = tuple(classes)
    if _supports_named_safe_globals():
        entries = [
            entry
            for cls in trusted
            for entry in (cls, (cls, f"{legacy_module}.{cls.__name__}"))
        ]
        with torch.serialization.safe_globals(entries):
            yield
        return

    with _ALIAS_LOCK:
        original_modules = tuple((cls, cls.__module__) for cls in trusted)
        try:
            for cls in trusted:
                cls.__module__ = legacy_module
            with torch.serialization.safe_globals(list(trusted)):
                yield
        finally:
            for cls, module in original_modules:
                cls.__module__ = module
