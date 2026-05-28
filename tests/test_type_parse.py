import dataclasses
import enum
import typing
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError, model_validator
from type_parse import create_parser
from type_parse.type_parse import is_dataclass_instance, is_dataclass_type, is_namedtuple_type

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
class WithNestedDefault:
    child: WithDefault = dataclasses.field(default_factory=WithDefault)


@dataclasses.dataclass
class PositivePoint:
    x: float
    y: float

    @model_validator(mode="after")
    def check_positive(self):
        if self.x <= 0 or self.y <= 0:
            raise ValueError("x and y must be positive")
        return self


@dataclasses.dataclass
class NegativePoint:
    x: float
    y: float

    @model_validator(mode="after")
    def check_negative(self):
        if self.x >= 0 or self.y >= 0:
            raise ValueError("x and y must be negative")
        return self


# ---------------------------------------------------------------------------
# 基本型
# ---------------------------------------------------------------------------


class TestBasicTypes:
    def test_int_passthrough(self):
        assert create_parser(int).validate_python(42) == 42

    def test_str_to_int(self):
        assert create_parser(int).validate_python("42") == 42

    def test_float_passthrough(self):
        assert create_parser(float).validate_python(3.14) == pytest.approx(3.14)

    def test_str_to_float(self):
        assert create_parser(float).validate_python("3.14") == pytest.approx(3.14)

    def test_invalid_raises(self):
        with pytest.raises(ValidationError):
            create_parser(int).validate_python("not-an-int")


# ---------------------------------------------------------------------------
# Enum（名前文字列パース）
# ---------------------------------------------------------------------------


class TestEnumParsing:
    def test_parse_by_name(self):
        assert create_parser(Color).validate_python("RED") is Color.RED

    def test_parse_by_name_green(self):
        assert create_parser(Color).validate_python("GREEN") is Color.GREEN

    def test_parse_instance_directly(self):
        assert create_parser(Color).validate_python(Color.BLUE) is Color.BLUE

    def test_invalid_name_raises(self):
        with pytest.raises(ValidationError):
            create_parser(Color).validate_python("YELLOW")


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------


class TestPathParsing:
    def test_no_root(self):
        assert create_parser(Path).validate_python("foo/bar") == Path("foo/bar")

    def test_with_root_resolves(self):
        root = Path("/tmp")
        result = create_parser(Path, root=root).validate_python("sub/file.txt")
        assert result == Path("/tmp/sub/file.txt")

    def test_absolute_path_with_root(self):
        root = Path("/tmp")
        result = create_parser(Path, root=root).validate_python("a")
        assert result == Path("/tmp/a").resolve()


# ---------------------------------------------------------------------------
# コレクション型
# ---------------------------------------------------------------------------


class TestCollections:
    def test_list_of_ints(self):
        assert create_parser(list[int]).validate_python([1, 2, 3]) == [1, 2, 3]

    def test_list_converts_elements(self):
        assert create_parser(list[int]).validate_python(["1", "2"]) == [1, 2]

    def test_list_invalid_raises(self):
        with pytest.raises(ValidationError):
            create_parser(list[int]).validate_python(["a", "b"])

    def test_set_of_ints(self):
        assert create_parser(set[int]).validate_python([1, 2, 3]) == {1, 2, 3}

    def test_set_deduplicates(self):
        assert create_parser(set[int]).validate_python([1, 1, 2]) == {1, 2}

    def test_dict_str_to_int(self):
        assert create_parser(dict[str, int]).validate_python({"a": 1}) == {"a": 1}

    def test_dict_value_conversion(self):
        assert create_parser(dict[str, int]).validate_python({"x": "10"}) == {"x": 10}

    def test_tuple_fixed(self):
        assert create_parser(tuple[int, str]).validate_python((1, "a")) == (1, "a")

    def test_tuple_variable_length(self):
        assert create_parser(tuple[int, ...]).validate_python([1, 2, 3]) == (1, 2, 3)

    def test_tuple_variable_converts_elements(self):
        assert create_parser(tuple[int, ...]).validate_python(["1", "2"]) == (1, 2)

    def test_nested_list_of_dicts(self):
        result = create_parser(list[dict[str, int]]).validate_python([{"a": 1}, {"b": "2"}])
        assert result == [{"a": 1}, {"b": 2}]

    def test_dict_of_list_of_ints(self):
        result = create_parser(dict[str, list[int]]).validate_python({"x": ["1", "2"]})
        assert result == {"x": [1, 2]}


# ---------------------------------------------------------------------------
# Union
# ---------------------------------------------------------------------------


class TestUnion:
    def test_none_type_in_union(self):
        assert create_parser(int | None).validate_python(None) is None

    def test_int_in_optional_union(self):
        assert create_parser(int | None).validate_python(42) == 42

    def test_all_fail_raises(self):
        with pytest.raises(ValidationError):
            create_parser(int | None).validate_python("not-a-number")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


class TestDataclassParsing:
    def test_parse_simple_dataclass(self):
        result = create_parser(Point).validate_python({"x": 1.0, "y": 2.0})
        assert result == Point(x=1.0, y=2.0)

    def test_parse_with_type_conversion(self):
        result = create_parser(Point).validate_python({"x": "3", "y": "4"})
        assert result == Point(x=3.0, y=4.0)

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            create_parser(Point).validate_python({"x": 1.0})

    def test_default_value_used(self):
        result = create_parser(WithDefault).validate_python({})
        assert result.value == 42

    def test_default_factory_used(self):
        result1 = create_parser(WithFactory).validate_python({})
        result2 = create_parser(WithFactory).validate_python({})
        assert result1.items == []
        assert result1.items is not result2.items

    def test_explicit_value_overrides_default(self):
        result = create_parser(WithDefault).validate_python({"value": 99})
        assert result.value == 99

    def test_nested_dataclass(self):
        result = create_parser(Nested).validate_python({"point": {"x": 1.0, "y": 2.0}})
        assert result == Nested(point=Point(x=1.0, y=2.0), label="default")

    def test_field_error_raises(self):
        with pytest.raises(ValidationError):
            create_parser(Point).validate_python({"x": "bad", "y": 1.0})

    def test_dataclass_instance_passed_directly(self):
        p = Point(x=1.0, y=2.0)
        result = create_parser(Point).validate_python(p)
        assert result == p

    def test_default_factory_returning_dataclass(self):
        result1 = create_parser(WithNestedDefault).validate_python({})
        result2 = create_parser(WithNestedDefault).validate_python({})
        assert result1.child == WithDefault(value=42)
        assert result1.child is not result2.child

    def test_collects_multiple_field_errors(self):
        with pytest.raises(ValidationError) as exc_info:
            create_parser(Point).validate_python({"x": "bad_x", "y": "bad_y"})
        errors = exc_info.value.errors()
        assert len(errors) == 2
        locs = {e["loc"] for e in errors}
        assert ("x",) in locs
        assert ("y",) in locs


# ---------------------------------------------------------------------------
# Enum フィールドを持つ Dataclass
# ---------------------------------------------------------------------------


class TestDataclassWithEnum:
    def test_enum_field_by_name(self):
        @dataclasses.dataclass
        class Config:
            color: Color

        result = create_parser(Config).validate_python({"color": "RED"})
        assert result.color is Color.RED

    def test_enum_field_by_instance(self):
        @dataclasses.dataclass
        class Config:
            color: Color

        result = create_parser(Config).validate_python({"color": Color.GREEN})
        assert result.color is Color.GREEN


# ---------------------------------------------------------------------------
# Path フィールドを持つ Dataclass（root 伝播）
# ---------------------------------------------------------------------------


class TestDataclassWithPath:
    def test_path_field_with_root(self):
        @dataclasses.dataclass
        class Config:
            path: Path
            name: str

        root = Path("/base")
        result = create_parser(Config, root=root).validate_python({"path": "foo/bar", "name": "x"})
        assert result.path == Path("/base/foo/bar")

    def test_path_field_without_root(self):
        @dataclasses.dataclass
        class Config:
            path: Path

        result = create_parser(Config).validate_python({"path": "foo/bar"})
        assert result.path == Path("foo/bar")

    def test_nested_dataclass_path_root_propagates(self):
        @dataclasses.dataclass
        class Inner:
            path: Path

        @dataclasses.dataclass
        class Outer:
            inner: Inner

        root = Path("/base")
        result = create_parser(Outer, root=root).validate_python({"inner": {"path": "foo"}})
        assert result.inner.path == Path("/base/foo").resolve()


# ---------------------------------------------------------------------------
# Post-init / Union dispatch
# ---------------------------------------------------------------------------


class TestUnionDispatch:
    def test_post_init_validation_failure(self):
        with pytest.raises(ValidationError):
            create_parser(PositivePoint).validate_python({"x": -1.0, "y": 2.0})

    def test_post_init_validation_success(self):
        result = create_parser(PositivePoint).validate_python({"x": 1.0, "y": 2.0})
        assert result.x == 1.0 and result.y == 2.0

    def test_union_dispatch_positive(self):
        ta = create_parser(PositivePoint | NegativePoint)
        result = ta.validate_python({"x": 1.0, "y": 2.0})
        assert type(result) is PositivePoint

    def test_union_dispatch_negative(self):
        ta = create_parser(PositivePoint | NegativePoint)
        result = ta.validate_python({"x": -1.0, "y": -2.0})
        assert type(result) is NegativePoint

    def test_union_all_fail_raises(self):
        with pytest.raises(ValidationError):
            create_parser(PositivePoint | NegativePoint).validate_python({"x": 1.0, "y": -2.0})

    def test_union_with_dict_config(self):
        @dataclasses.dataclass
        class WithTuple:
            vap_bins: tuple[float, float]

            @model_validator(mode="after")
            def check(self):
                return self

        raw: DictConfig = OmegaConf.create({"vap_bins": [0.1, 0.9]})
        result = create_parser(WithTuple | None).validate_python(raw)
        assert result == WithTuple(vap_bins=(0.1, 0.9))


# ---------------------------------------------------------------------------
# NamedTuple
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


class TestNamedTupleParsing:
    def test_parse_ok(self):
        result = create_parser(NTPoint).validate_python({"x": 1.0, "y": 2.0})
        assert result == NTPoint(x=1.0, y=2.0)

    def test_type_conversion(self):
        result = create_parser(NTPoint).validate_python({"x": "1.5", "y": "2.5"})
        assert result == NTPoint(x=1.5, y=2.5)

    def test_default_used_when_field_absent(self):
        result = create_parser(NTWithDefault).validate_python({})
        assert result == NTWithDefault(value=42, label="hello")

    def test_default_overridden_when_field_present(self):
        result = create_parser(NTWithDefault).validate_python({"value": 99})
        assert result == NTWithDefault(value=99, label="hello")

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            create_parser(NTPoint).validate_python({"x": 1.0})

    def test_invalid_value_raises(self):
        with pytest.raises(ValidationError):
            create_parser(NTPoint).validate_python({"x": "bad", "y": 2.0})

    def test_collects_multiple_errors(self):
        with pytest.raises(ValidationError) as exc_info:
            create_parser(NTPoint).validate_python({"x": "bad_x", "y": "bad_y"})
        assert exc_info.value.error_count() == 2

    def test_nested_namedtuple(self):
        result = create_parser(NTNested).validate_python({"point": {"x": 1.0, "y": 2.0}})
        assert result == NTNested(point=NTPoint(x=1.0, y=2.0), tag="default")

    def test_omegaconf_dictconfig_input(self):
        raw: DictConfig = OmegaConf.create({"x": 1.0, "y": 2.0})
        result = create_parser(NTPoint).validate_python(raw)
        assert result == NTPoint(x=1.0, y=2.0)

    def test_namedtuple_instance_passed_directly(self):
        p = NTPoint(x=1.0, y=2.0)
        result = create_parser(NTPoint).validate_python(p)
        assert result == p

    def test_default_returning_namedtuple(self):
        result = create_parser(NTWithNestedDefault).validate_python({})
        assert result.nested == NTWithDefault(value=42, label="hello")


# ---------------------------------------------------------------------------
# Guard 関数
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


class TestIsNamedTupleType:
    def test_namedtuple_class(self):
        assert is_namedtuple_type(NTPoint) is True

    def test_plain_tuple_is_false(self):
        assert is_namedtuple_type(tuple) is False

    def test_dataclass_is_false(self):
        assert is_namedtuple_type(Point) is False

    def test_instance_is_false(self):
        assert is_namedtuple_type(NTPoint(1.0, 2.0)) is False


