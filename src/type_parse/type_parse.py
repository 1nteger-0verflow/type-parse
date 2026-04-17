import dataclasses
import enum
import types
import typing
from abc import ABC, abstractmethod
from collections import abc
from pathlib import Path
from typing import Any

_MISSING = object()


class _DataclassInstance(typing.Protocol):
    __dataclass_fields__: typing.ClassVar[dict[str, dataclasses.Field]]


class _NamedTupleInstance(typing.Protocol):
    _fields: tuple[str, ...]
    _field_defaults: dict[str, Any]


def is_dataclass_type(obj: object) -> typing.TypeGuard[type[_DataclassInstance]]:
    return dataclasses.is_dataclass(obj) and isinstance(obj, type)


def is_dataclass_instance(obj: object) -> typing.TypeGuard[_DataclassInstance]:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def is_namedtuple_type(obj: object) -> typing.TypeGuard[type[_NamedTupleInstance]]:
    return isinstance(obj, type) and issubclass(obj, tuple) and hasattr(obj, "_fields")


@dataclasses.dataclass(frozen=True)
class ParseErr:
    names: tuple[str, ...]
    expected_type: type | None = None
    actual_value: object = dataclasses.field(default_factory=lambda: _MISSING)


class TypeParseError(Exception):
    def __init__(self, errors: list[ParseErr], msg: str = ""):
        self.errors = errors
        self.msg = msg

    def __str__(self):
        lines = [self.msg] if self.msg else []
        for err in self.errors:
            path = ".".join(err.names) or "<root>"
            if err.actual_value is not _MISSING:
                actual = type(err.actual_value).__name__
                if err.expected_type is not None:
                    lines.append(f"  {path}: cannot convert {actual!r} to {err.expected_type.__name__!r}")
                else:
                    lines.append(f"  {path}: failed to parse {actual!r}")
            else:
                lines.append(f"  {path}")
        return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class Ok[T]:
    value: T

    def unwrap(self) -> T:
        return self.value

    def expect(self, msg: str = "") -> T:  # noqa: ARG002
        return self.value


@dataclasses.dataclass(frozen=True)
class Err[E]:
    error: E

    def expect(self, msg: str = "") -> typing.Never:
        raise TypeParseError(self.error, msg)  # type: ignore[arg-type]


type Result[T, E] = Ok[T] | Err[E]


class Parser[T](ABC):
    @abstractmethod
    def parse(self, value: Any, *, names: tuple[str, ...] = ()) -> Result[T, list[ParseErr]]:  # noqa: ANN401
        """Primary implementation: full scan, returns Result (no exceptions)."""
        ...


class TypeParser[T](Parser[T]):
    def __init__(self, type_: type[T]):
        super().__init__()
        self._type = type_

    def parse(self, value: Any, *, names: tuple[str, ...] = ()) -> Result[T, list[ParseErr]]:  # noqa: ANN401
        try:
            if isinstance(value, self._type):
                return Ok(value)
            return Ok(self._type(value))
        except Exception:  # noqa: BLE001
            return Err([ParseErr(names=names, expected_type=self._type, actual_value=value)])


class EnumParser[T: enum.Enum](Parser[T]):
    def __init__(self, type_: type[T]):
        super().__init__()
        self._type = type_

    def parse(self, value: Any, *, names: tuple[str, ...] = ()) -> Result[T, list[ParseErr]]:  # noqa: ANN401
        try:
            if isinstance(value, self._type):
                return Ok(value)
            return Ok(self._type[value])
        except Exception:  # noqa: BLE001
            return Err([ParseErr(names=names, expected_type=self._type, actual_value=value)])


class PathParser(Parser[Path]):
    def __init__(self, root: Path | None):
        super().__init__()
        self._root = root

    def parse(self, value: Any, *, names: tuple[str, ...] = ()) -> Result[Path, list[ParseErr]]:  # noqa: ANN401
        try:
            if self._root is None:
                return Ok(Path(value))
            return Ok((self._root / Path(value)).resolve())
        except Exception:  # noqa: BLE001
            return Err([ParseErr(names=names, actual_value=value)])


class UnionParser[T](Parser[T]):
    def __init__(self, value_parsers: tuple[Parser, ...]):
        super().__init__()
        self._value_parsers = value_parsers

    def parse(self, value: Any, *, names: tuple[str, ...] = ()) -> Result[Any, list[ParseErr]]:  # noqa: ANN401
        results = []
        for parser in self._value_parsers:
            r = parser.parse(value, names=names)
            if isinstance(r, Ok):
                results.append(r)

        if not results:
            return Err([ParseErr(names=names, actual_value=value)])

        no_changes = [r for r in results if type(r.value) is type(value) and r.value == value]
        return no_changes[0] if no_changes else results[0]


class ListParser[T](Parser[list[T]]):
    def __init__(self, value_parser: Parser[T]):
        super().__init__()
        self._value_parser = value_parser

    def _iter_parse(self, value: Any, *, names: tuple[str, ...] = ()):  # noqa: ANN401
        for i, v in enumerate(value):
            yield i, self._value_parser.parse(v, names=(*names, str(i)))

    def parse(self, value: Any, *, names: tuple[str, ...] = ()) -> Result[list[T], list[ParseErr]]:  # noqa: ANN401
        if not isinstance(value, abc.Iterable):
            return Err([ParseErr(names=names, actual_value=value)])
        errors, parsed = [], []
        for _i, result in self._iter_parse(value, names=names):
            if isinstance(result, Err):
                errors.extend(result.error)
            else:
                parsed.append(result.value)
        if errors:
            return Err(errors)
        return Ok(parsed)


class SetParser[T](Parser[set[T]]):
    def __init__(self, value_parser: Parser[T]):
        super().__init__()
        self._value_parser = value_parser

    def _iter_parse(self, value: Any, *, names: tuple[str, ...] = ()):  # noqa: ANN401
        for i, v in enumerate(value):
            yield i, self._value_parser.parse(v, names=(*names, str(i)))

    def parse(self, value: Any, *, names: tuple[str, ...] = ()) -> Result[set[T], list[ParseErr]]:  # noqa: ANN401
        if not isinstance(value, abc.Iterable):
            return Err([ParseErr(names=names, actual_value=value)])
        errors: list[ParseErr] = []
        parsed: set = set()
        for _i, result in self._iter_parse(value, names=names):
            if isinstance(result, Err):
                errors.extend(result.error)
            else:
                parsed.add(result.value)
        if errors:
            return Err(errors)
        return Ok(parsed)


class DictParser[K, V](Parser[dict[K, V]]):
    def __init__(self, key_parser: Parser[K], value_parser: Parser[V]):
        super().__init__()
        self._key_parser = key_parser
        self._value_parser = value_parser

    def _iter_parse(self, value: Any, *, names: tuple[str, ...] = ()):  # noqa: ANN401
        for k, v in value.items():
            yield k, self._key_parser.parse(k, names=names), self._value_parser.parse(v, names=(*names, str(k)))

    def parse(self, value: Any, *, names: tuple[str, ...] = ()) -> Result[dict[K, V], list[ParseErr]]:  # noqa: ANN401
        if not isinstance(value, abc.Mapping):
            return Err([ParseErr(names=names, actual_value=value)])
        errors: list[ParseErr] = []
        parsed: dict = {}
        for _k, key_result, val_result in self._iter_parse(value, names=names):
            for result in (key_result, val_result):
                if isinstance(result, Err):
                    errors.extend(result.error)
            if isinstance(key_result, Ok) and isinstance(val_result, Ok):
                parsed[key_result.value] = val_result.value
        if errors:
            return Err(errors)
        return Ok(parsed)


class TupleParser[*Ts](Parser[tuple[*Ts]]):
    def __init__(self, value_parsers: tuple[Parser | types.EllipsisType, ...]):
        super().__init__()
        self._forward_parsers, self._ellipsis_parser, self._backward_parsers = self._split_parsers(value_parsers)

    def _iter_parse(self, value: Any, *, names: tuple[str, ...] = ()):  # noqa: ANN401
        if not isinstance(value, abc.Sequence) or isinstance(value, (str, bytes)):
            yield None, Err([ParseErr(names=names, actual_value=value)])
            return
        n_values = len(value)
        n_forward = len(self._forward_parsers)
        n_backward = len(self._backward_parsers)

        if n_values < n_forward + n_backward:
            yield None, Err([ParseErr(names=names, actual_value=value)])
            return
        if n_values > n_forward + n_backward and self._ellipsis_parser is None:
            yield None, Err([ParseErr(names=names, actual_value=value)])
            return

        for i, (p, v) in enumerate(zip(self._forward_parsers, value[:n_forward])):
            yield i, p.parse(v, names=(*names, str(i)))

        if n_values > n_forward + n_backward:
            for i, v in enumerate(value[n_forward : n_values - n_backward], start=n_forward):
                yield i, self._ellipsis_parser.parse(v, names=(*names, str(i)))  # type: ignore[union-attr]

        if n_backward > 0:
            for i, (p, v) in enumerate(zip(self._backward_parsers, value[-n_backward:]), start=n_values - n_backward):
                yield i, p.parse(v, names=(*names, str(i)))

    def parse(self, value: Any, *, names: tuple[str, ...] = ()) -> Result[tuple, list[ParseErr]]:  # noqa: ANN401
        errors: list[ParseErr] = []
        parsed = []
        for i, result in self._iter_parse(value, names=names):
            if i is None:
                return result  # type: ignore[return-value]  # length error → Err[list[ParseErr]]
            if isinstance(result, Err):
                errors.extend(result.error)
            else:
                parsed.append(result.value)  # type: ignore[union-attr]
        if errors:
            return Err(errors)
        return Ok(tuple(parsed))

    def _split_parsers(
        self, parsers: tuple[Parser | types.EllipsisType, ...]
    ) -> tuple[tuple[Parser, ...], Parser | None, tuple[Parser, ...]]:
        idx_iter = (i for i, c in enumerate(parsers) if c is Ellipsis)
        idx_ellipsis = next(idx_iter, None)

        if idx_ellipsis is None:
            return typing.cast("tuple[Parser, ...]", parsers), None, ()

        if idx_ellipsis == 0:
            msg = "Ellipsis cannot be the first element"
            raise RuntimeError(msg)

        if next(idx_iter, None) is not None:
            msg = "cannot have multi Ellipsis"
            raise RuntimeError(msg)

        forward_parsers = parsers[: idx_ellipsis - 1]
        ellipsis_parser = parsers[idx_ellipsis - 1]
        backward_parsers = parsers[idx_ellipsis + 1 :]

        return (
            typing.cast("tuple[Parser, ...]", forward_parsers),
            typing.cast("Parser", ellipsis_parser),
            typing.cast("tuple[Parser, ...]", backward_parsers),
        )


class DataclassParser[T: _DataclassInstance](Parser[T]):
    def __init__(self, cls: type[T], value_parsers: dict[str, Parser], defaults: dict[str, Any] | None = None):
        super().__init__()
        self._cls = cls
        self._value_parsers = value_parsers
        self._defaults = {} if defaults is None else defaults

    def _iter_parse(self, value: Any, *, names: tuple[str, ...] = ()):  # noqa: ANN401
        for n, p in self._value_parsers.items():
            if is_dataclass_instance(value):
                v = getattr(value, n)
            elif isinstance(value, abc.Mapping) and n in value:
                v = value[n]
            else:
                v = self.get_default(n)
            yield n, p.parse(v, names=(*names, n))

    def parse(self, value: Any, *, names: tuple[str, ...] = ()) -> Result[T, list[ParseErr]]:  # noqa: ANN401
        if not (is_dataclass_instance(value) or isinstance(value, abc.Mapping)):
            return Err([ParseErr(names=names, actual_value=value)])
        errors: list[ParseErr] = []
        parsed: dict[str, Any] = {}
        for n, result in self._iter_parse(value, names=names):
            if isinstance(result, Err):
                errors.extend(result.error)
            else:
                parsed[n] = result.value
        if errors:
            return Err(errors)
        try:
            return Ok(self._cls(**parsed))
        except Exception:  # noqa: BLE001
            return Err([ParseErr(names=names, actual_value=value)])

    def get_default(self, name: str):
        if name not in self._defaults:
            return _MISSING

        if isinstance(self._defaults[name], abc.Callable):
            return self._defaults[name]()
        return self._defaults[name]


class NamedTupleParser[T: _NamedTupleInstance](Parser[T]):
    def __init__(self, cls: type[T], value_parsers: dict[str, Parser], defaults: dict[str, Any]):
        super().__init__()
        self._cls = cls
        self._value_parsers = value_parsers
        self._defaults = defaults

    def _iter_parse(self, value: Any, *, names: tuple[str, ...] = ()):  # noqa: ANN401
        for n, p in self._value_parsers.items():
            if isinstance(value, self._cls):
                v = getattr(value, n)
            elif isinstance(value, abc.Mapping) and n in value:
                v = value[n]
            else:
                v = self._defaults.get(n, _MISSING)
            yield n, p.parse(v, names=(*names, n))

    def parse(self, value: Any, *, names: tuple[str, ...] = ()) -> Result[T, list[ParseErr]]:  # noqa: ANN401
        if not isinstance(value, (self._cls, abc.Mapping)):
            return Err([ParseErr(names=names, actual_value=value)])
        errors: list[ParseErr] = []
        parsed: dict[str, Any] = {}
        for n, result in self._iter_parse(value, names=names):
            if isinstance(result, Err):
                errors.extend(result.error)
            else:
                parsed[n] = result.value
        if errors:
            return Err(errors)
        try:
            return Ok(self._cls(**parsed))
        except Exception:  # noqa: BLE001
            return Err([ParseErr(names=names, actual_value=value)])


def _fetch_defaults(fields: tuple[dataclasses.Field, ...]):
    defaults = {}
    for field in fields:
        if field.default is not dataclasses.MISSING:
            if isinstance(field.default, abc.Callable):
                msg = "Unsupported default value"
                raise TypeError(msg)
            defaults[field.name] = field.default
            continue

        if field.default_factory is not dataclasses.MISSING:
            if not isinstance(field.default_factory, abc.Callable):
                msg = "Unsupported default factory"
                raise TypeError(msg)
            defaults[field.name] = field.default_factory
            continue

    return defaults


@typing.overload
def create_parser[T: _DataclassInstance](t: type[T], *, root: Path | None = None) -> DataclassParser[T]: ...


@typing.overload
def create_parser(t: type[Path], *, root: Path | None = None) -> PathParser: ...


@typing.overload
def create_parser[T: enum.Enum](t: type[T], *, root: Path | None = None) -> EnumParser[T]: ...


@typing.overload
def create_parser[T](t: type[list[T]], *, root: Path | None = None) -> ListParser[T]: ...


@typing.overload
def create_parser[T](t: type[set[T]], *, root: Path | None = None) -> SetParser[T]: ...


@typing.overload
def create_parser[K, V](t: type[dict[K, V]], *, root: Path | None = None) -> DictParser[K, V]: ...


@typing.overload
def create_parser[T: _NamedTupleInstance](t: type[T], *, root: Path | None = None) -> NamedTupleParser[T]: ...


@typing.overload
def create_parser[*Ts](t: type[tuple[*Ts]], *, root: Path | None = None) -> TupleParser[*Ts]: ...


@typing.overload
def create_parser[T](t: type[T], *, root: Path | None = None) -> UnionParser[T]: ...


@typing.overload
def create_parser[T](t: type[T], *, root: Path | None = None) -> TypeParser[T]: ...


def _create_parser_for_concrete_type(t: type, root: Path | None) -> Parser:
    if issubclass(t, Path):
        return PathParser(root)
    if issubclass(t, enum.Enum):
        return EnumParser(t)
    return TypeParser(t)


def _create_parser_for_generic(t: types.GenericAlias, type_args: tuple, root: Path | None) -> Parser:
    type_origin = typing.get_origin(t)
    if type_origin is list:
        return ListParser(create_parser(type_args[0], root=root))
    if type_origin is set:
        return SetParser(create_parser(type_args[0], root=root))
    if type_origin is dict:
        return DictParser(create_parser(type_args[0], root=root), create_parser(type_args[1], root=root))
    if type_origin is tuple:
        return TupleParser(tuple(create_parser(a, root=root) if a is not Ellipsis else Ellipsis for a in type_args))
    raise RuntimeError


def create_parser(t, *, root: Path | None = None):  # type: ignore[misc]  # noqa: ANN001
    if is_dataclass_type(t):
        type_hints = typing.get_type_hints(t)
        return DataclassParser(
            t,
            value_parsers={f.name: create_parser(type_hints[f.name], root=root) for f in dataclasses.fields(t)},
            defaults=_fetch_defaults(dataclasses.fields(t)),
        )

    if is_namedtuple_type(t):
        type_hints = typing.get_type_hints(t)
        return NamedTupleParser(
            t,
            value_parsers={name: create_parser(type_hints[name], root=root) for name in t._fields},
            defaults=dict(t._field_defaults),
        )

    type_args = typing.get_args(t)

    if isinstance(t, type):
        return _create_parser_for_concrete_type(t, root)

    if isinstance(t, types.GenericAlias):
        return _create_parser_for_generic(t, type_args, root)

    if isinstance(t, types.UnionType):
        return UnionParser(tuple(create_parser(a, root=root) for a in type_args))

    raise NotImplementedError
