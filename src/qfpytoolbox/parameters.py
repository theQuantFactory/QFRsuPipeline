"""Parameters interface — mirrors ``MyJuliaToolbox.jl/src/parameters.jl``.

Provides an abstract base class ``iParameters`` for configuration objects with
JSON serialization/deserialization and media-aware read/write helpers.
"""

from __future__ import annotations

import json
import os
from abc import ABC
from typing import Any, TypeVar

__all__ = [
    "iParameters",
    "parameters_from_dict",
    "parameters_from_json",
    "read_parameters",
    "write_parameters",
    "nonpersisted_fields",
]

T = TypeVar("T", bound="iParameters")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class iParameters(ABC):  # noqa: B024
    """Abstract base type for parameter/configuration objects.

    Concrete subclasses should be plain Python classes (or dataclasses) with
    typed fields.  :func:`parameters_from_dict` constructs them from a
    ``dict`` by matching keys to ``__init__`` parameters.

    Example
    -------
    .. code-block:: python

        from dataclasses import dataclass
        from qfpytoolbox.parameters import iParameters

        @dataclass
        class ModelParams(iParameters):
            learning_rate: float
            epochs: int
            batch_size: int
    """


# ---------------------------------------------------------------------------
# nonpersisted_fields (dataset integration)
# ---------------------------------------------------------------------------


def nonpersisted_fields(cls: type) -> tuple[str, ...]:
    """Return field names that ``write_dataset`` should skip for ``cls``.

    Override per subclass to mark fields that should not be persisted.  The
    default implementation returns an empty tuple.

    Example
    -------
    .. code-block:: python

        from qfpytoolbox.parameters import nonpersisted_fields
        from qfpytoolbox.dataset import iDataSet

        class MyDataSet(iDataSet):
            data: pd.DataFrame
            _cached: pd.DataFrame  # not persisted

        def nonpersisted_fields_my(cls):
            return ("_cached",)
    """
    return ()


# ---------------------------------------------------------------------------
# parameters_from_dict / parameters_from_json
# ---------------------------------------------------------------------------


def parameters_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Construct an ``iParameters`` subclass from a dictionary.

    Keys are matched to ``__init__`` parameters.  Nested dicts that correspond
    to ``iParameters`` subtype fields are recursively constructed.

    Raises
    ------
    ValueError
        If a required key is missing or an unexpected key is found.
    """
    return _construct_from_dict(cls, data)


def parameters_from_json(cls: type[T], path: str) -> T:
    """Load a JSON file and construct an ``iParameters`` subclass."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return parameters_from_dict(cls, data)


# ---------------------------------------------------------------------------
# write_parameters
# ---------------------------------------------------------------------------


def write_parameters(
    dest: Any,
    params: iParameters,
    file_name: str | None = None,
    relative_path: str = "",
    *,
    overwrite: bool = False,
    pretty: bool = True,
) -> None:
    """Write ``params`` as JSON.

    Parameters
    ----------
    dest:
        - ``str`` path (``*.json``) — write directly.
        - :class:`~qfpytoolbox.io.media.iSourceMedia` — resolve location from
          the media source.
    params:
        Parameter object to serialise.
    file_name:
        When ``dest`` is a media source, the JSON file name.
    relative_path:
        Optional sub-directory under the media source root.
    overwrite:
        Allow overwriting an existing file (default ``False``).
    pretty:
        Pretty-print JSON output with 2-space indent (default ``True``).
    """
    if isinstance(dest, str):
        _write_to_path(dest, params, overwrite=overwrite, pretty=pretty)
        return

    from qfpytoolbox.io.media import iSourceMedia  # noqa: PLC0415

    if isinstance(dest, iSourceMedia):
        if file_name is None:
            raise ValueError("file_name is required when dest is an iSourceMedia")
        _write_via_media(dest, file_name, params, relative_path, overwrite=overwrite, pretty=pretty)
        return

    raise TypeError(f"Unsupported dest type for write_parameters: {type(dest).__name__!r}")


def _write_to_path(path: str, params: iParameters, *, overwrite: bool, pretty: bool) -> None:
    if not path.lower().endswith(".json"):
        raise ValueError(f"Parameter file must have a .json extension: {path!r}")
    if os.path.isfile(path) and not overwrite:
        raise FileExistsError(f"Parameter file already exists: {path!r}. Pass overwrite=True to overwrite.")
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    payload = _to_json_value(params)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(payload, f, indent=2)
        else:
            json.dump(payload, f)


def _write_via_media(
    source: Any,
    file_name: str,
    params: iParameters,
    relative_path: str,
    *,
    overwrite: bool,
    pretty: bool,
) -> None:
    from qfpytoolbox.io.media import DatabaseMedia, FileSystemMedia, SQLDumpMedia  # noqa: PLC0415

    if isinstance(source, FileSystemMedia):
        root = source.path
        full_path = _resolve_location(root, file_name, relative_path)
        _write_to_path(full_path, params, overwrite=overwrite, pretty=pretty)
    elif isinstance(source, DatabaseMedia):
        if source.parameters_locator is None:
            raise ValueError("DatabaseMedia requires 'parameters_locator' to write parameter files.")
        full_path = _resolve_location(source.parameters_locator, file_name, relative_path)
        _write_to_path(full_path, params, overwrite=overwrite, pretty=pretty)
    elif isinstance(source, SQLDumpMedia):
        raise TypeError("SQLDumpMedia is read-only; cannot write parameters to it.")
    else:
        raise TypeError(f"write_parameters is not implemented for {type(source).__name__!r}")


# ---------------------------------------------------------------------------
# read_parameters
# ---------------------------------------------------------------------------


def read_parameters(
    cls: type[T],
    source: Any,
    file_name: str,
    relative_path: str = "",
) -> T:
    """Read and construct an ``iParameters`` subclass from a media source.

    Parameters
    ----------
    cls:
        Target parameter class.
    source:
        :class:`~qfpytoolbox.io.media.iSourceMedia` or ``str`` path.
    file_name:
        JSON file name.
    relative_path:
        Optional sub-directory under the media source root.
    """
    if isinstance(source, str):
        full = _resolve_location(source, file_name, relative_path)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"Parameter file not found: {full!r}")
        return parameters_from_json(cls, full)

    from qfpytoolbox.io.media import DatabaseMedia, FileSystemMedia, SQLDumpMedia  # noqa: PLC0415

    if isinstance(source, FileSystemMedia):
        full = _resolve_location(source.path, file_name, relative_path)
    elif isinstance(source, (DatabaseMedia, SQLDumpMedia)):
        locator = source.parameters_locator
        if locator is None:
            raise ValueError(f"{type(source).__name__} requires 'parameters_locator' to read parameters.")
        full = _resolve_location(locator, file_name, relative_path)
    else:
        raise TypeError(f"read_parameters is not implemented for {type(source).__name__!r}")

    if not os.path.isfile(full):
        raise FileNotFoundError(f"Parameter file not found: {full!r}")
    return parameters_from_json(cls, full)


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _to_json_value(value: Any) -> Any:
    if isinstance(value, iParameters):
        return {k: _to_json_value(v) for k, v in _iter_fields(value)}
    if isinstance(value, dict):
        return {str(k): _to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(v) for v in value]
    return value


def _iter_fields(obj: Any):
    """Yield (name, value) pairs from a dataclass or plain iParameters object."""
    import dataclasses  # noqa: PLC0415

    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            yield f.name, getattr(obj, f.name)
    else:
        for name in vars(obj):
            if not name.startswith("_"):
                yield name, getattr(obj, name)


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _construct_from_dict(cls: type, data: dict[str, Any]) -> Any:
    import inspect

    # Determine expected keys from __init__ signature (skip 'self')
    try:
        sig = inspect.signature(cls.__init__)
        expected = {
            name
            for name, p in sig.parameters.items()
            if name != "self"
            and p.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
    except (ValueError, TypeError):
        expected = None

    if expected is not None:
        extra = set(data) - expected
        if extra:
            raise ValueError(f"Unexpected keys for {cls.__name__!r}: {sorted(extra)!r}")
        missing = expected - set(data)
        # check defaults
        sig2 = inspect.signature(cls.__init__)
        missing_no_default = {k for k in missing if sig2.parameters[k].default is inspect.Parameter.empty}
        if missing_no_default:
            raise ValueError(f"Missing keys for {cls.__name__!r}: {sorted(missing_no_default)!r}")

    # Get type hints to support nested iParameters
    hints: dict[str, Any] = {}
    try:
        import typing  # noqa: PLC0415

        hints = typing.get_type_hints(cls)
    except Exception:  # noqa: BLE001
        pass

    kwargs: dict[str, Any] = {}
    for key, val in data.items():
        hint = hints.get(key)
        kwargs[key] = _convert_value(hint, val)

    return cls(**kwargs)


def _convert_value(hint: Any, value: Any) -> Any:  # noqa: PLR0911
    if value is None:
        return None

    if hint is None:
        return value

    import dataclasses  # noqa: PLC0415, E401
    import typing

    # Union / Optional
    origin = getattr(hint, "__origin__", None)
    if origin is typing.Union:
        args = hint.__args__
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _convert_value(non_none[0], value)
        return value

    # list[X]
    if origin is list:
        args = getattr(hint, "__args__", None)
        elem_type = args[0] if args else None
        return [_convert_value(elem_type, v) for v in value]

    # iParameters subtype
    if isinstance(hint, type) and issubclass(hint, iParameters) and isinstance(value, dict):
        return _construct_from_dict(hint, value)

    # dataclass
    if isinstance(hint, type) and dataclasses.is_dataclass(hint) and isinstance(value, dict):
        return _construct_from_dict(hint, value)

    try:
        return hint(value)
    except (TypeError, ValueError):
        return value


# ---------------------------------------------------------------------------
# Location resolver
# ---------------------------------------------------------------------------


def _resolve_location(root: str, file_name: str, relative_path: str) -> str:
    if not relative_path:
        return os.path.join(root, file_name)
    return os.path.join(root, relative_path, file_name)
