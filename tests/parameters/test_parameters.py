"""Tests for iParameters — JSON configuration management."""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

from qfpytoolbox.parameters import (
    iParameters,
    parameters_from_dict,
    parameters_from_json,
    read_parameters,
    write_parameters,
)

# ---------------------------------------------------------------------------
# Fixtures / sample types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SimpleParams(iParameters):
    learning_rate: float
    epochs: int
    name: str


@dataclasses.dataclass
class NestedParams(iParameters):
    model: SimpleParams
    batch_size: int


@dataclasses.dataclass
class OptionalParams(iParameters):
    value: float
    label: str = "default"


# ---------------------------------------------------------------------------
# parameters_from_dict
# ---------------------------------------------------------------------------


class TestParametersFromDict:
    def test_basic(self):
        data = {"learning_rate": 0.01, "epochs": 10, "name": "test"}
        p = parameters_from_dict(SimpleParams, data)
        assert p.learning_rate == 0.01
        assert p.epochs == 10
        assert p.name == "test"

    def test_type_coercion(self):
        data = {"learning_rate": "0.001", "epochs": "5", "name": "x"}
        p = parameters_from_dict(SimpleParams, data)
        assert isinstance(p.learning_rate, float)
        assert isinstance(p.epochs, int)

    def test_nested(self):
        data = {
            "model": {"learning_rate": 0.1, "epochs": 3, "name": "m"},
            "batch_size": 32,
        }
        p = parameters_from_dict(NestedParams, data)
        assert isinstance(p.model, SimpleParams)
        assert p.model.epochs == 3
        assert p.batch_size == 32

    def test_extra_key_raises(self):
        data = {"learning_rate": 0.01, "epochs": 5, "name": "x", "extra": "bad"}
        with pytest.raises(ValueError, match="Unexpected keys"):
            parameters_from_dict(SimpleParams, data)

    def test_missing_key_raises(self):
        data = {"learning_rate": 0.01}
        with pytest.raises((ValueError, TypeError)):
            parameters_from_dict(SimpleParams, data)


# ---------------------------------------------------------------------------
# parameters_from_json
# ---------------------------------------------------------------------------


class TestParametersFromJson:
    def test_round_trip(self, tmp_path):
        p = str(tmp_path / "params.json")
        data = {"learning_rate": 0.01, "epochs": 5, "name": "lr"}
        with open(p, "w") as f:
            json.dump(data, f)
        result = parameters_from_json(SimpleParams, p)
        assert result.learning_rate == 0.01


# ---------------------------------------------------------------------------
# write_parameters / read_parameters
# ---------------------------------------------------------------------------


class TestWriteReadParameters:
    def test_write_and_read_back(self, tmp_path):
        p = str(tmp_path / "params.json")
        params = SimpleParams(learning_rate=0.001, epochs=20, name="exp1")
        write_parameters(p, params)
        assert os.path.isfile(p)
        result = parameters_from_json(SimpleParams, p)
        assert result.learning_rate == 0.001
        assert result.epochs == 20
        assert result.name == "exp1"

    def test_overwrite_false_raises(self, tmp_path):
        p = str(tmp_path / "params.json")
        params = SimpleParams(learning_rate=0.01, epochs=1, name="a")
        write_parameters(p, params)
        with pytest.raises(FileExistsError, match="already exists"):
            write_parameters(p, params)

    def test_overwrite_true(self, tmp_path):
        p = str(tmp_path / "params.json")
        params = SimpleParams(learning_rate=0.01, epochs=1, name="a")
        write_parameters(p, params)
        write_parameters(p, params, overwrite=True)  # no raise

    def test_non_json_extension_raises(self, tmp_path):
        with pytest.raises(ValueError, match=".json"):
            write_parameters(str(tmp_path / "params.txt"), SimpleParams(0.01, 1, "x"))

    def test_creates_parent_dirs(self, tmp_path):
        p = str(tmp_path / "sub" / "dir" / "params.json")
        params = SimpleParams(0.01, 1, "x")
        write_parameters(p, params)
        assert os.path.isfile(p)

    def test_nested_write_read(self, tmp_path):
        p = str(tmp_path / "nested.json")
        inner = SimpleParams(0.1, 3, "inner")
        params = NestedParams(model=inner, batch_size=64)
        write_parameters(p, params)
        result = parameters_from_json(NestedParams, p)
        assert result.batch_size == 64
        assert result.model.name == "inner"

    def test_read_via_filesystem_media(self, tmp_path):
        from qfpytoolbox.io.media import FileSystemMedia

        p = str(tmp_path / "params.json")
        params = SimpleParams(0.05, 2, "fs")
        write_parameters(p, params)
        media = FileSystemMedia(str(tmp_path))
        result = read_parameters(SimpleParams, media, "params.json")
        assert result.epochs == 2

    def test_read_file_not_found_raises(self, tmp_path):
        from qfpytoolbox.io.media import FileSystemMedia

        media = FileSystemMedia(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            read_parameters(SimpleParams, media, "missing.json")

    def test_pretty_json_format(self, tmp_path):
        p = str(tmp_path / "params.json")
        params = SimpleParams(0.01, 1, "x")
        write_parameters(p, params, pretty=True)
        with open(p) as f:
            content = f.read()
        assert "\n" in content  # pretty printed

    def test_compact_json_format(self, tmp_path):
        p = str(tmp_path / "params.json")
        params = SimpleParams(0.01, 1, "x")
        write_parameters(p, params, pretty=False)
        with open(p) as f:
            content = f.read()
        assert content.count("\n") <= 1
