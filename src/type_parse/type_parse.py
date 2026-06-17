import dataclasses
import enum
import types
import typing
from collections import abc
from pathlib import Path
from typing import Any, TypeGuard, cast, overload

from pydantic import TypeAdapter as _TypeAdapter
from pydantic import ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError


def is_dataclass_type(obj: object) -> bool:
    return dataclasses.is_dataclass(obj) and isinstance(obj, type)


def is_dataclass_instance(obj: object) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


class _NamedTupleType(typing.Protocol):
    _fields: tuple[str, ...]


def is_namedtuple_type(obj: object) -> TypeGuard[type[_NamedTupleType]]:
    return isinstance(obj, type) and issubclass(obj, tuple) and hasattr(obj, "_fields")


def _preprocess_data(value: Any, type_: Any, root: Path | None) -> Any:
    # Enum: by name string
    if isinstance(type_, type) and issubclass(type_, enum.Enum):
        if isinstance(value, str):
            try:
                return type_[value]
            except KeyError:
                valid = [e.name for e in type_]
                msg = f"'{value}' is not a valid {type_.__name__} name. Valid names: {valid}"
                raise ValueError(msg) from None
        return value

    # Path: resolve relative to root
    if type_ is Path and root is not None:
        return (root / Path(value)).resolve()

    origin = typing.get_origin(type_)
    args = typing.get_args(type_)

    if origin is types.UnionType or origin is typing.Union:
        if value is None and type(None) in args:
            return None
        for t in (a for a in args if a is not type(None)):
            try:
                preprocessed = _preprocess_data(value, t, root)
                _TypeAdapter(t).validate_python(preprocessed)
                return preprocessed
            except Exception:  # noqa: BLE001, S112
                continue
        return value

    # list
    if origin is list and args and isinstance(value, list):
        return [_preprocess_data(item, args[0], root) for item in value]

    # set
    if origin is set and args and isinstance(value, (list, set)):
        return [_preprocess_data(item, args[0], root) for item in value]

    # tuple
    if origin is tuple and args and isinstance(value, (list, tuple)):
        if len(args) == 2 and args[1] is Ellipsis:
            return [_preprocess_data(item, args[0], root) for item in value]
        return [_preprocess_data(item, t, root) for item, t in zip(value, args)]

    # dict
    if origin is dict and args and isinstance(value, abc.Mapping):
        k_type, v_type = args
        return {_preprocess_data(k, k_type, root): _preprocess_data(v, v_type, root) for k, v in value.items()}

    # Mapping -> dataclass または NamedTuple
    if (is_dataclass_type(type_) or is_namedtuple_type(type_)) and isinstance(value, abc.Mapping):
        hints = typing.get_type_hints(type_)
        return {k: _preprocess_data(v, hints.get(k, type(v)), root) for k, v in value.items()}

    # dataclass
    if is_dataclass_type(type_) and is_dataclass_instance(value):
        hints = typing.get_type_hints(type_)
        return {k: _preprocess_data(getattr(value, k), hints.get(k, Any), root) for k in hints}

    # NamedTuple
    if is_namedtuple_type(type_) and isinstance(value, tuple):
        hints = typing.get_type_hints(type_)
        return {k: _preprocess_data(getattr(value, k), hints.get(k, Any), root) for k in type_._fields}

    return value


class TypeAdapter[T]:
    @overload
    def __init__(self, type_: type[T], root: Path | None = None) -> None: ...
    @overload
    def __init__(self, type_: Any, root: Path | None = None) -> None: ...
    def __init__(self, type_: Any, root: Path | None = None) -> None:
        self._type = type_
        self._inner = cast("_TypeAdapter[Any]", _TypeAdapter(type_))
        self._root = root

    def validate_python(self, value: Any, **kwargs: Any) -> T:
        try:
            preprocessed = _preprocess_data(value, self._type, self._root)
        except ValueError as exc:
            title = getattr(self._type, "__name__", repr(self._type))
            raise ValidationError.from_exception_data(
                title=title,
                input_type="python",
                line_errors=[
                    InitErrorDetails(
                        type=PydanticCustomError("value_error", "{msg}", {"msg": str(exc)}), loc=(), input=value
                    )
                ],
            ) from None
        return self._inner.validate_python(preprocessed, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
