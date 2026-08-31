from contextlib import contextmanager

import pytest

from tmol.database.scoring import _serialization


class Example:
    pass


@contextmanager
def _recording_safe_globals(record, entries):
    record.append(tuple(entries))
    yield


def test_aliased_safe_globals_torch_2_5(monkeypatch):
    observed = []
    original_module = Example.__module__
    monkeypatch.setattr(_serialization.torch, "__version__", "2.5.1+cu124")
    monkeypatch.setattr(
        _serialization.torch.serialization,
        "safe_globals",
        lambda entries: _recording_safe_globals(observed, entries),
    )

    with _serialization.aliased_safe_globals([Example], "legacy.module"):
        assert Example.__module__ == "legacy.module"

    assert Example.__module__ == original_module
    assert observed == [(Example,)]


def test_aliased_safe_globals_torch_2_6(monkeypatch):
    observed = []
    original_module = Example.__module__
    monkeypatch.setattr(_serialization.torch, "__version__", "2.6.0")
    monkeypatch.setattr(
        _serialization.torch.serialization,
        "safe_globals",
        lambda entries: _recording_safe_globals(observed, entries),
    )

    with _serialization.aliased_safe_globals([Example], "legacy.module"):
        assert Example.__module__ == original_module

    assert observed == [
        (Example, (Example, "legacy.module.Example")),
    ]


def test_aliased_safe_globals_rejects_unparseable_torch_version(monkeypatch):
    monkeypatch.setattr(_serialization.torch, "__version__", "development")
    with pytest.raises(RuntimeError, match="Cannot parse PyTorch version"):
        with _serialization.aliased_safe_globals([Example], "legacy.module"):
            pass
