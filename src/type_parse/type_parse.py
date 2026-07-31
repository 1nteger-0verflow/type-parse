import dataclasses
import enum
import types
import typing
from collections import abc
from pathlib import Path
from typing import Any, TypeGuard, cast, overload

from pydantic import TypeAdapter, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError


class _DataclassInstance(typing.Protocol):
    __dataclass_fields__: typing.ClassVar[dict[str, dataclasses.Field[Any]]]


def is_dataclass_type(obj: object) -> TypeGuard[type[_DataclassInstance]]:
    return dataclasses.is_dataclass(obj) and isinstance(obj, type)


def is_dataclass_instance(obj: object) -> TypeGuard[_DataclassInstance]:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


class _NamedTupleType(typing.Protocol):
    _fields: tuple[str, ...]


def is_namedtuple_type(obj: object) -> TypeGuard[type[_NamedTupleType]]:
    return isinstance(obj, type) and issubclass(obj, tuple) and hasattr(obj, "_fields")


def _is_sequence_value(value: object) -> bool:
    return isinstance(value, abc.Sequence) and not isinstance(value, (str, bytes))


type _Loc = tuple[int | str, ...]


@dataclasses.dataclass(frozen=True)
class _Ok[T]:
    value: T


@dataclasses.dataclass(frozen=True)
class _Err:
    errors: list[InitErrorDetails]


type _Result[T] = _Ok[T] | _Err


def _sequence[T](results: abc.Sequence[_Result[T]]) -> _Result[list[T]]:
    errors = [e for r in results if isinstance(r, _Err) for e in r.errors]
    if errors:
        return _Err(errors)
    return _Ok([r.value for r in results if isinstance(r, _Ok)])


def _sequence_mapping[K, V](results: abc.Mapping[K, _Result[V]]) -> _Result[dict[K, V]]:
    errors = [e for r in results.values() if isinstance(r, _Err) for e in r.errors]
    if errors:
        return _Err(errors)
    return _Ok({k: r.value for k, r in results.items() if isinstance(r, _Ok)})


def _preprocess_data(value: Any, type_: Any, root: Path | None, *, loc: _Loc = ()) -> _Result[Any]:
    # Enum: by name string
    if isinstance(type_, type) and issubclass(type_, enum.Enum):
        if isinstance(value, str):
            try:
                return _Ok(type_[value])
            except KeyError:
                valid = [e.name for e in type_]
                msg = f"'{value}' is not a valid {type_.__name__} name. Valid names: {valid}"
                return _Err(
                    [
                        InitErrorDetails(
                            type=PydanticCustomError("value_error", "{msg}", {"msg": msg}), loc=loc, input=value
                        )
                    ]
                )
        return _Ok(value)

    # Path: resolve relative to root
    if type_ is Path and root is not None:
        return _Ok((root / Path(value)).resolve())

    origin = typing.get_origin(type_)
    args = typing.get_args(type_)

    if origin is types.UnionType or origin is typing.Union:
        if value is None and type(None) in args:
            return _Ok(None)
        for t in (a for a in args if a is not type(None)):
            result = _preprocess_data(value, t, root, loc=loc)
            if isinstance(result, _Err):
                continue
            try:
                TypeAdapter(t).validate_python(result.value)
            except Exception:  # noqa: BLE001
                continue
            return result
        return _Ok(value)

    # list
    if origin is list and args and _is_sequence_value(value):
        return _sequence([_preprocess_data(item, args[0], root, loc=(*loc, i)) for i, item in enumerate(value)])

    # set
    if origin is set and args and (_is_sequence_value(value) or isinstance(value, abc.Set)):
        return _sequence([_preprocess_data(item, args[0], root, loc=(*loc, i)) for i, item in enumerate(value)])

    # tuple
    if origin is tuple and args and _is_sequence_value(value):
        if len(args) == 2 and args[1] is Ellipsis:
            return _sequence([_preprocess_data(item, args[0], root, loc=(*loc, i)) for i, item in enumerate(value)])
        return _sequence(
            [_preprocess_data(item, t, root, loc=(*loc, i)) for i, (item, t) in enumerate(zip(value, args))]
        )

    # dict
    if origin is dict and args and isinstance(value, abc.Mapping):
        k_type, v_type = args
        key_results = [_preprocess_data(k, k_type, root, loc=loc) for k in value]
        val_results = [_preprocess_data(v, v_type, root, loc=(*loc, k)) for k, v in value.items()]
        errors = [e for r in (*key_results, *val_results) if isinstance(r, _Err) for e in r.errors]
        if errors:
            return _Err(errors)
        keys = [r.value for r in key_results if isinstance(r, _Ok)]
        vals = [r.value for r in val_results if isinstance(r, _Ok)]
        return _Ok(dict(zip(keys, vals)))

    # Mapping -> dataclass または NamedTuple
    if (is_dataclass_type(type_) or is_namedtuple_type(type_)) and isinstance(value, abc.Mapping):
        hints = typing.get_type_hints(type_)
        return _sequence_mapping(
            {k: _preprocess_data(v, hints.get(k, type(v)), root, loc=(*loc, k)) for k, v in value.items()}
        )

    # dataclass
    if is_dataclass_type(type_) and is_dataclass_instance(value):
        hints = typing.get_type_hints(type_)
        init_names = {f.name for f in dataclasses.fields(type_) if f.init}
        return _sequence_mapping(
            {
                k: _preprocess_data(getattr(value, k), hints.get(k, Any), root, loc=(*loc, k))
                for k in hints
                if k in init_names
            }
        )

    # NamedTuple
    if is_namedtuple_type(type_) and isinstance(value, tuple):
        hints = typing.get_type_hints(type_)
        return _sequence_mapping(
            {k: _preprocess_data(getattr(value, k), hints.get(k, Any), root, loc=(*loc, k)) for k in type_._fields}
        )

    return _Ok(value)


type _FieldLoc = tuple[str | int, ...]


@dataclasses.dataclass(frozen=True)
class FieldError:
    msg: str
    error_type: str
    input: Any


def errors_by_field(e: ValidationError) -> dict[_FieldLoc, list[FieldError]]:
    result: dict[_FieldLoc, list[FieldError]] = {}
    for err in e.errors(include_url=False):
        loc: _FieldLoc = tuple(err["loc"])
        result.setdefault(loc, []).append(
            FieldError(msg=err["msg"], error_type=err["type"], input=err.get("input"))
        )
    return result


class TypeParser[T]:
    @overload
    def __init__(self, type_: type[T], root: Path | None = None) -> None: ...
    @overload
    def __init__(self, type_: Any, root: Path | None = None) -> None: ...
    def __init__(self, type_: Any, root: Path | None = None) -> None:
        self._type = type_
        self._inner = cast("TypeAdapter[Any]", TypeAdapter(type_))
        self._root = root

    def parse(self, value: Any, **kwargs: Any) -> T:
        result = _preprocess_data(value, self._type, self._root)
        if isinstance(result, _Err):
            title = getattr(self._type, "__name__", repr(self._type))
            raise ValidationError.from_exception_data(title=title, input_type="python", line_errors=result.errors)
        return self._inner.validate_python(result.value, **kwargs)
