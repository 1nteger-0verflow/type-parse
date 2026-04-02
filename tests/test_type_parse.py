import dataclasses
import enum
import typing
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf
from type_parse.type_parse import (
    DataclassParser,
    DictParser,
    EnumParser,
    Err,
    ListParser,
    NamedTupleParser,
    Ok,
    ParseErr,
    PathParser,
    SetParser,
    TupleParser,
    TypeParseError,
    TypeParser,
    UnionParser,
    _fetch_defaults,
    create_parser,
    is_dataclass_instance,
    is_dataclass_type,
    is_namedtuple_type,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class Color(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


@dataclasses.dataclass
class Point:
    x: float
    y: float


@dataclasses.dataclass
class WithDefault:
    value: int = 42


@dataclasses.dataclass
class WithFactory:
    items: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Nested:
    point: Point
    label: str = "default"


@dataclasses.dataclass
class PositivePoint:
    x: float
    y: float

    def __post_init__(self):
        if self.x <= 0 or self.y <= 0:
            raise ValueError("x and y must be positive")


@dataclasses.dataclass
class NegativePoint:
    x: float
    y: float

    def __post_init__(self):
        if self.x >= 0 or self.y >= 0:
            raise ValueError("x and y must be negative")


@dataclasses.dataclass
class WithNestedDefault:
    child: WithDefault = dataclasses.field(default_factory=WithDefault)


# ---------------------------------------------------------------------------
# TypeParser
# ---------------------------------------------------------------------------


class TestTypeParser:
    def test_already_correct_type(self):
        parser = TypeParser(int)
        assert parser.parse(42).expect("") == 42

    def test_converts_str_to_int(self):
        parser = TypeParser(int)
        assert parser.parse("10").expect("") == 10

    def test_converts_to_float(self):
        parser = TypeParser(float)
        assert parser.parse("3.14").expect("") == pytest.approx(3.14)

    def test_invalid_conversion_raises_typeparseerror(self):
        parser = TypeParser(int)
        with pytest.raises(TypeParseError):
            parser.parse("not-an-int").expect("")

    def test_names_propagated_in_error(self):
        parser = TypeParser(int)
        with pytest.raises(TypeParseError) as exc_info:
            parser.parse("bad", names=("root", "child")).expect("")
        assert exc_info.value.errors[0].names == ("root", "child")


# ---------------------------------------------------------------------------
# EnumParser
# ---------------------------------------------------------------------------


class TestEnumParser:
    def test_parse_enum_instance_directly(self):
        parser = EnumParser(Color)
        assert parser.parse(Color.RED).expect("") is Color.RED

    def test_parse_by_name(self):
        parser = EnumParser(Color)
        assert parser.parse("GREEN").expect("") is Color.GREEN

    def test_invalid_name_raises_typeparseerror(self):
        parser = EnumParser(Color)
        with pytest.raises(TypeParseError):
            parser.parse("YELLOW").expect("")


# ---------------------------------------------------------------------------
# PathParser
# ---------------------------------------------------------------------------


class TestPathParser:
    def test_no_root(self):
        parser = PathParser(root=None)
        result = parser.parse("foo/bar").expect("")
        assert result == Path("foo/bar")

    def test_with_root_resolves(self):
        root = Path("/tmp")
        parser = PathParser(root=root)
        result = parser.parse("sub/file.txt").expect("")
        assert result == Path("/tmp/sub/file.txt")

    def test_absolute_path_with_root(self):
        root = Path("/tmp")
        parser = PathParser(root=root)
        result = parser.parse("a").expect("")
        assert result == Path("/tmp/a").resolve()

    def test_parse_dict_returns_err(self):
        result = PathParser(root=None).parse({"key": "val"})
        assert isinstance(result, Err)


# ---------------------------------------------------------------------------
# UnionParser
# ---------------------------------------------------------------------------


class TestUnionParser:
    def test_no_change_preferred(self):
        # value is already int — should prefer the no-change result
        parser = UnionParser((TypeParser(int), TypeParser(str)))
        result = parser.parse(5).expect("")
        assert result == 5
        assert isinstance(result, int)

    def test_conversion_when_needed(self):
        parser = UnionParser((TypeParser(int), TypeParser(str)))
        result = parser.parse("hello").expect("")
        # "hello" can't be int; str succeeds without change
        assert result == "hello"
        assert isinstance(result, str)

    def test_all_fail_raises_typeparseerror(self):
        parser = UnionParser((TypeParser(int),))
        with pytest.raises(TypeParseError):
            parser.parse("not-a-number").expect("")

    def test_none_type_in_union(self):
        assert create_parser(int | None).parse(None).expect("") is None

    def test_int_in_optional_union(self):
        assert create_parser(int | None).parse(42).expect("") == 42

    def test_returns_first_converted_value_when_no_unchanged(self):
        # float | int: parse("5") — float("5")=5.0 and int("5")=5 both convert,
        # neither equals the original str "5", so values[0] (float) is returned
        parser = UnionParser((TypeParser(float), TypeParser(int)))
        result = parser.parse("5").expect("")
        assert result == 5.0
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# ListParser
# ---------------------------------------------------------------------------


class TestListParser:
    def test_parse_list_of_ints(self):
        parser = ListParser(TypeParser(int))
        assert parser.parse([1, 2, 3]).expect("") == [1, 2, 3]

    def test_converts_elements(self):
        parser = ListParser(TypeParser(int))
        assert parser.parse(["1", "2", "3"]).expect("") == [1, 2, 3]

    def test_element_error_has_index_in_name(self):
        parser = ListParser(TypeParser(int))
        with pytest.raises(TypeParseError) as exc_info:
            parser.parse([1, "bad", 3]).expect("")
        # "bad" fails at index 1
        assert any("1" in e.names for e in exc_info.value.errors)

    def test_empty_list(self):
        parser = ListParser(TypeParser(int))
        assert parser.parse([]).expect("") == []


# ---------------------------------------------------------------------------
# SetParser
# ---------------------------------------------------------------------------


class TestSetParser:
    def test_parse_set(self):
        parser = SetParser(TypeParser(int))
        result = parser.parse([1, 2, 3]).expect("")
        assert result == {1, 2, 3}

    def test_deduplicates(self):
        parser = SetParser(TypeParser(int))
        result = parser.parse([1, 1, 2]).expect("")
        assert result == {1, 2}

    def test_element_error_returns_err(self):
        result = SetParser(TypeParser(int)).parse([1, "bad", 3])
        assert isinstance(result, Err)
        assert len(result.error) == 1


# ---------------------------------------------------------------------------
# DictParser
# ---------------------------------------------------------------------------


class TestDictParser:
    def test_parse_str_to_int_dict(self):
        parser = DictParser(TypeParser(str), TypeParser(int))
        result = parser.parse({"a": 1, "b": 2}).expect("")
        assert result == {"a": 1, "b": 2}

    def test_value_conversion(self):
        parser = DictParser(TypeParser(str), TypeParser(int))
        result = parser.parse({"x": "10"}).expect("")
        assert result == {"x": 10}

    def test_empty_dict(self):
        parser = DictParser(TypeParser(str), TypeParser(int))
        assert parser.parse({}).expect("") == {}

    def test_key_failure_returns_err(self):
        result = DictParser(TypeParser(int), TypeParser(str)).parse({"bad": "v"})
        assert isinstance(result, Err)
        assert len(result.error) == 1

    def test_value_failure_returns_err(self):
        result = DictParser(TypeParser(str), TypeParser(int)).parse({"k": "bad"})
        assert isinstance(result, Err)
        assert len(result.error) == 1

    def test_key_and_value_failure_returns_two_errors(self):
        result = DictParser(TypeParser(int), TypeParser(int)).parse({"bad": "bad"})
        assert isinstance(result, Err)
        assert len(result.error) == 2


# ---------------------------------------------------------------------------
# TupleParser
# ---------------------------------------------------------------------------


class TestTupleParser:
    def test_fixed_length(self):
        parser = TupleParser((TypeParser(int), TypeParser(str)))
        result = parser.parse((1, "hello")).expect("")
        assert result == (1, "hello")

    def test_fixed_with_conversion(self):
        parser = TupleParser((TypeParser(int), TypeParser(float)))
        result = parser.parse(("3", "1.5")).expect("")
        assert result == (3, 1.5)

    def test_variable_length_with_ellipsis(self):
        # tuple[int, ...]  →  parsers=(TypeParser(int), Ellipsis)
        parser = TupleParser((TypeParser(int), Ellipsis))
        result = parser.parse((1, 2, 3, 4)).expect("")
        assert result == (1, 2, 3, 4)

    def test_variable_length_empty_middle(self):
        parser = TupleParser((TypeParser(int), Ellipsis))
        result = parser.parse((42,)).expect("")
        assert result == (42,)

    def test_too_few_elements_raises(self):
        # fixed (int, str) requires exactly 2
        parser = TupleParser((TypeParser(int), TypeParser(str)))
        with pytest.raises(TypeParseError):
            parser.parse((1,)).expect("")

    def test_too_many_elements_without_ellipsis_raises(self):
        # Fixed-length (int, str) but 3 elements supplied → Err
        parser = TupleParser((TypeParser(int), TypeParser(str)))
        with pytest.raises(TypeParseError):
            parser.parse((1, "a", "extra")).expect("")

    def test_split_parsers_ellipsis_at_index_0_raises(self):
        # Ellipsis at position 0 is invalid
        with pytest.raises(RuntimeError):
            TupleParser((Ellipsis, TypeParser(int)))

    def test_split_parsers_multi_ellipsis_raises(self):
        # Two Ellipsis markers are invalid
        with pytest.raises(RuntimeError, match="cannot have multi Ellipsis"):
            TupleParser((TypeParser(int), Ellipsis, TypeParser(str), Ellipsis))

    def test_backward_parsers_with_elements(self):
        # tuple[str, int, ..., float]: forward=(str,), ellipsis=int, backward=(float,)
        parser = TupleParser((TypeParser(str), TypeParser(int), Ellipsis, TypeParser(float)))
        result = parser.parse(("a", 1, 2, 3, 9.9)).expect("")
        assert result == ("a", 1, 2, 3, 9.9)

    def test_backward_parsers_zero_middle_elements(self):
        # Same parser, but middle section is empty
        parser = TupleParser((TypeParser(str), TypeParser(int), Ellipsis, TypeParser(float)))
        result = parser.parse(("a", 1, 9.9)).expect("")
        assert result == ("a", 1, 9.9)


# ---------------------------------------------------------------------------
# DataclassParser
# ---------------------------------------------------------------------------


class TestDataclassParser:
    def test_parse_simple_dataclass(self):
        parser = create_parser(Point)
        result = parser.parse({"x": 1.0, "y": 2.0}).expect("")
        assert result == Point(x=1.0, y=2.0)

    def test_parse_with_type_conversion(self):
        parser = create_parser(Point)
        result = parser.parse({"x": "3", "y": "4"}).expect("")
        assert result == Point(x=3.0, y=4.0)

    def test_missing_field_returns_err(self):
        parser = create_parser(Point)
        result = parser.parse({"x": 1.0})
        # y is missing — parse() collects all errors
        assert isinstance(result, Err)
        assert isinstance(result.error, list)
        assert any("y" in e.names for e in result.error)

    def test_default_value_used(self):
        parser = create_parser(WithDefault)
        result = parser.parse({}).expect("")
        assert result.value == 42

    def test_default_factory_used(self):
        # This exercises the bug we fixed: default_factory must work
        parser = create_parser(WithFactory)
        result = parser.parse({}).expect("")
        assert result.items == []
        # Each call should return a new list (factory called each time)
        result2 = parser.parse({}).expect("")
        assert result.items is not result2.items

    def test_explicit_value_overrides_default(self):
        parser = create_parser(WithDefault)
        result = parser.parse({"value": 99}).expect("")
        assert result.value == 99

    def test_nested_dataclass(self):
        parser = create_parser(Nested)
        result = parser.parse({"point": {"x": 1.0, "y": 2.0}}).expect("")
        assert result == Nested(point=Point(x=1.0, y=2.0), label="default")

    def test_field_error_includes_field_name(self):
        parser = create_parser(Point)
        with pytest.raises(TypeParseError) as exc_info:
            parser.parse({"x": "bad", "y": 1.0}).expect("")
        assert any("x" in e.names for e in exc_info.value.errors)

    def test_dataclass_instance_passed_directly(self):
        p = Point(x=1.0, y=2.0)
        result = create_parser(Point).parse(p).expect("")
        assert result is p

    def test_default_factory_returning_dataclass(self):
        result = create_parser(WithNestedDefault).parse({}).expect("")
        assert result.child == WithDefault(value=42)
        result2 = create_parser(WithNestedDefault).parse({}).expect("")
        assert result.child is not result2.child


# ---------------------------------------------------------------------------
# _fetch_defaults
# ---------------------------------------------------------------------------


class TestFetchDefaults:
    def test_plain_default(self):
        defaults = _fetch_defaults(dataclasses.fields(WithDefault))
        assert defaults == {"value": 42}

    def test_default_factory_stored_as_callable(self):
        defaults = _fetch_defaults(dataclasses.fields(WithFactory))
        assert "items" in defaults
        assert callable(defaults["items"])
        assert defaults["items"]() == []

    def test_required_field_not_in_defaults(self):
        defaults = _fetch_defaults(dataclasses.fields(Point))
        assert defaults == {}

    def test_callable_as_plain_default_raises(self):
        # A field whose default is a callable object (not default_factory) should raise
        @dataclasses.dataclass
        class BadCallableDefault:
            fn: object = dataclasses.field(default=lambda: None)

        with pytest.raises(TypeError):
            _fetch_defaults(dataclasses.fields(BadCallableDefault))


# ---------------------------------------------------------------------------
# create_parser — factory dispatching
# ---------------------------------------------------------------------------


class TestCreateParser:
    def test_creates_type_parser_for_int(self):
        assert isinstance(create_parser(int), TypeParser)

    def test_creates_enum_parser(self):
        assert isinstance(create_parser(Color), EnumParser)

    def test_creates_path_parser(self):
        assert isinstance(create_parser(Path), PathParser)

    def test_creates_list_parser(self):
        assert isinstance(create_parser(list[int]), ListParser)

    def test_creates_set_parser(self):
        assert isinstance(create_parser(set[str]), SetParser)

    def test_creates_dict_parser(self):
        assert isinstance(create_parser(dict[str, int]), DictParser)

    def test_creates_tuple_parser(self):
        assert isinstance(create_parser(tuple[int, str]), TupleParser)

    def test_creates_union_parser(self):
        assert isinstance(create_parser(int | str), UnionParser)

    def test_creates_dataclass_parser(self):
        assert isinstance(create_parser(Point), DataclassParser)

    def test_path_parser_propagates_root(self):
        root = Path("/some/root")
        parser = create_parser(Path, root=root)
        assert isinstance(parser, PathParser)
        result = parser.parse("file.txt").expect("")
        assert result == Path("/some/root/file.txt")

    def test_unsupported_generic_alias_raises(self):
        # frozenset[int] is a GenericAlias but not list/set/dict/tuple
        with pytest.raises(RuntimeError):
            create_parser(frozenset[int])

    def test_unsupported_type_raises_not_implemented(self):
        # typing.Union is a _GenericAlias (not types.UnionType)
        with pytest.raises(NotImplementedError):
            create_parser(typing.Union[int, str])

    def test_nested_list_of_dicts_parse(self):
        result = create_parser(list[dict[str, int]]).parse([{"a": 1}, {"b": "2"}]).expect("")
        assert result == [{"a": 1}, {"b": 2}]

    def test_dict_of_list_of_ints_parse(self):
        result = create_parser(dict[str, list[int]]).parse({"x": ["1", "2"]}).expect("")
        assert result == {"x": [1, 2]}


# ---------------------------------------------------------------------------
# is_dataclass_instance / is_dataclass_type
# ---------------------------------------------------------------------------


class TestDataclassGuards:
    def test_instance_true_for_dataclass_instance(self):
        assert is_dataclass_instance(Point(1.0, 2.0)) is True

    def test_instance_false_for_dataclass_class(self):
        assert is_dataclass_instance(Point) is False

    def test_instance_false_for_plain_object(self):
        assert is_dataclass_instance(42) is False

    def test_type_true_for_dataclass_class(self):
        assert is_dataclass_type(Point) is True

    def test_type_false_for_instance(self):
        assert is_dataclass_type(Point(1.0, 2.0)) is False


# ---------------------------------------------------------------------------
# TypeParseError
# ---------------------------------------------------------------------------


class TestTypeParseError:
    def test_str_representation(self):
        err = TypeParseError([ParseErr(names=("a", "b", "c"))], "")
        assert "a.b.c" in str(err)

    def test_empty_names_shows_root(self):
        err = TypeParseError([ParseErr(names=())], "")
        assert "<root>" in str(err)

    def test_is_exception(self):
        assert isinstance(TypeParseError([ParseErr(names=())], ""), Exception)

    def test_str_with_expected_type_and_actual_value(self):
        err = TypeParseError([ParseErr(names=("x", "y"), expected_type=int, actual_value="bad")], "")
        msg = str(err)
        assert "x.y" in msg
        assert "int" in msg
        assert "str" in msg

    def test_str_without_optional_fields(self):
        # expected_type only, no actual_value → just the path, no conversion message
        err = TypeParseError([ParseErr(names=("a",), expected_type=int)], "")
        s = str(err)
        assert "a" in s
        assert "cannot convert" not in s
        assert "failed to parse" not in s

    def test_str_with_only_actual_value(self):
        err = TypeParseError([ParseErr(names=("a",), actual_value=42)], "")
        assert "a" in str(err)
        assert "int" in str(err)

    def test_msg_included_in_str(self):
        err = TypeParseError([ParseErr(names=("a",))], "something went wrong")
        assert "something went wrong" in str(err)

    def test_multiple_errors_all_in_str(self):
        err = TypeParseError([ParseErr(names=("x",)), ParseErr(names=("y",))], "")
        s = str(err)
        assert "x" in s and "y" in s

    def test_empty_errors_list(self):
        err = TypeParseError([], "only message")
        assert str(err) == "only message"

    def test_parse_err_actual_value(self):
        parser = TypeParser(int)
        result = parser.parse("not_a_number", names=("field",))
        assert isinstance(result, Err)
        assert isinstance(result.error, list)
        assert result.error[0].actual_value == "not_a_number"
        with pytest.raises(TypeParseError) as exc_info:
            result.expect("test")
        assert "field" in str(exc_info.value)
        assert "'str'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Result types: Ok / Err / ParseErr / unwrap / expect
# ---------------------------------------------------------------------------


class TestResultTypes:
    def test_ok_unwrap(self):
        assert Ok(42).unwrap() == 42

    def test_ok_expect(self):
        assert Ok(42).expect("unused msg") == 42

    def test_err_expect_raises(self):
        errs = [ParseErr(names=("x",), expected_type=int, actual_value="bad")]
        with pytest.raises(TypeParseError) as exc_info:
            Err(errs).expect("failed")
        assert exc_info.value.errors[0].names == ("x",)
        assert exc_info.value.msg == "failed"


# ---------------------------------------------------------------------------
# parse: multi-error collection
# ---------------------------------------------------------------------------


class TestParseAll:
    def test_dataclass_parse_collects_all_errors(self):
        parser = create_parser(Point)
        result = parser.parse({"x": "bad_x", "y": "bad_y"})
        assert isinstance(result, Err)
        assert isinstance(result.error, list)
        assert len(result.error) == 2
        field_names = {name for e in result.error for name in e.names}
        assert "x" in field_names
        assert "y" in field_names

    def test_dataclass_parse_ok(self):
        parser = create_parser(Point)
        result = parser.parse({"x": 1.0, "y": 2.0})
        assert isinstance(result, Ok)
        assert result.value == Point(1.0, 2.0)

    def test_list_parse_collects_all_errors(self):
        parser = ListParser(TypeParser(int))
        result = parser.parse(["bad1", 2, "bad3"])
        assert isinstance(result, Err)
        assert isinstance(result.error, list)
        assert len(result.error) == 2

    def test_nested_dataclass_parse_collects_errors(self):
        parser = create_parser(Nested)
        result = parser.parse({"point": {"x": "bad", "y": "bad"}, "label": 123})
        assert isinstance(result, Err)
        assert isinstance(result.error, list)
        # x and y errors from nested Point, label converts fine (123 → "123")
        assert len(result.error) == 2
        field_names = {name for e in result.error for name in e.names}
        assert "x" in field_names and "y" in field_names


# ---------------------------------------------------------------------------
# __post_init__ exception handling
# ---------------------------------------------------------------------------


class TestPostInit:
    def test_post_init_exception_returns_err(self):
        result = create_parser(PositivePoint).parse({"x": -1.0, "y": 2.0})
        assert isinstance(result, Err)

    def test_post_init_success(self):
        result = create_parser(PositivePoint).parse({"x": 1.0, "y": 2.0}).expect("")
        assert result == PositivePoint(x=1.0, y=2.0)

    def test_union_dispatch_via_post_init(self):
        parser = create_parser(PositivePoint | NegativePoint)
        assert parser.parse({"x": 1.0, "y": 2.0}).expect("") == PositivePoint(x=1.0, y=2.0)
        assert parser.parse({"x": -1.0, "y": -2.0}).expect("") == NegativePoint(x=-1.0, y=-2.0)

    def test_union_all_post_init_fail_returns_err(self):
        result = create_parser(PositivePoint | NegativePoint).parse({"x": 1.0, "y": -2.0})
        assert isinstance(result, Err)

    def test_union_with_dict_config_no_exception(self):
        # UnionParser の等値比較で DictConfig.__eq__ が呼ばれないことを確認
        # (tuple フィールドを含む dataclass と DictConfig の比較で ValidationError が出ていた)
        @dataclasses.dataclass
        class WithTuple:
            vap_bins: tuple[float, float]

            def __post_init__(self):
                pass  # 常に成功

        raw: DictConfig = OmegaConf.create({"vap_bins": [0.1, 0.9]})
        result = create_parser(WithTuple | None).parse(raw)
        assert isinstance(result, Ok)
        assert result.value == WithTuple(vap_bins=(0.1, 0.9))


# ---------------------------------------------------------------------------
# NamedTupleParser
# ---------------------------------------------------------------------------


class NTPoint(typing.NamedTuple):
    x: float
    y: float


class NTWithDefault(typing.NamedTuple):
    value: int = 42
    label: str = "hello"


class NTNested(typing.NamedTuple):
    point: NTPoint
    tag: str = "default"


class NTWithNestedDefault(typing.NamedTuple):
    nested: NTWithDefault = NTWithDefault()


class TestIsNamedTupleType:
    def test_namedtuple_class(self):
        assert is_namedtuple_type(NTPoint) is True

    def test_plain_tuple_is_false(self):
        assert is_namedtuple_type(tuple) is False

    def test_dataclass_is_false(self):
        assert is_namedtuple_type(Point) is False

    def test_instance_is_false(self):
        assert is_namedtuple_type(NTPoint(1.0, 2.0)) is False


class TestNamedTupleParser:
    def test_create_parser_returns_namedtuple_parser(self):
        assert isinstance(create_parser(NTPoint), NamedTupleParser)

    def test_parse_ok(self):
        result = create_parser(NTPoint).parse({"x": 1.0, "y": 2.0})
        assert isinstance(result, Ok)
        assert result.value == NTPoint(x=1.0, y=2.0)

    def test_type_conversion(self):
        result = create_parser(NTPoint).parse({"x": "1.5", "y": "2.5"})
        assert isinstance(result, Ok)
        assert result.value == NTPoint(x=1.5, y=2.5)

    def test_default_used_when_field_absent(self):
        result = create_parser(NTWithDefault).parse({})
        assert isinstance(result, Ok)
        assert result.value == NTWithDefault(value=42, label="hello")

    def test_default_overridden_when_field_present(self):
        result = create_parser(NTWithDefault).parse({"value": 99})
        assert isinstance(result, Ok)
        assert result.value == NTWithDefault(value=99, label="hello")

    def test_missing_required_field_returns_err(self):
        result = create_parser(NTPoint).parse({"x": 1.0})
        assert isinstance(result, Err)
        field_names = {name for e in result.error for name in e.names}
        assert "y" in field_names

    def test_invalid_value_returns_err(self):
        result = create_parser(NTPoint).parse({"x": "bad", "y": 2.0})
        assert isinstance(result, Err)
        assert result.error[0].expected_type is float

    def test_collects_all_errors(self):
        result = create_parser(NTPoint).parse({"x": "bad_x", "y": "bad_y"})
        assert isinstance(result, Err)
        assert len(result.error) == 2
        field_names = {name for e in result.error for name in e.names}
        assert "x" in field_names and "y" in field_names

    def test_error_names_propagated(self):
        result = create_parser(NTPoint).parse({"x": "bad", "y": 2.0}, names=("root",))
        assert isinstance(result, Err)
        assert result.error[0].names == ("root", "x")

    def test_nested_namedtuple(self):
        result = create_parser(NTNested).parse({"point": {"x": 1.0, "y": 2.0}})
        assert isinstance(result, Ok)
        assert result.value == NTNested(point=NTPoint(x=1.0, y=2.0), tag="default")

    def test_nested_error_collected(self):
        result = create_parser(NTNested).parse({"point": {"x": "bad", "y": "bad"}})
        assert isinstance(result, Err)
        assert len(result.error) == 2

    def test_omegaconf_dictconfig_input(self):
        raw: DictConfig = OmegaConf.create({"x": 1.0, "y": 2.0})
        result = create_parser(NTPoint).parse(raw)
        assert isinstance(result, Ok)
        assert result.value == NTPoint(x=1.0, y=2.0)

    def test_namedtuple_instance_passed_directly(self):
        p = NTPoint(x=1.0, y=2.0)
        result = create_parser(NTPoint).parse(p).expect("")
        assert result is p

    def test_default_returning_namedtuple(self):
        result = create_parser(NTWithNestedDefault).parse({}).expect("")
        assert result.nested == NTWithDefault(value=42, label="hello")
