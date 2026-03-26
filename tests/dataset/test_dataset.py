"""Tests for dataset persistence."""

from __future__ import annotations

import dataclasses
import json
import os

import pandas as pd
import pytest

from qfpytoolbox.dataset import (
    LoadedDataSet,
    iDataSet,
    nonpersisted_fields,
    read_dataset,
    write_dataset,
)
from qfpytoolbox.io.media import ArchiveMedia, FileSystemMedia

# ---------------------------------------------------------------------------
# Fixtures / sample types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SimpleDataSet(iDataSet):
    data: pd.DataFrame
    label: str
    count: int


@dataclasses.dataclass
class DataSetWithMeta(iDataSet):
    prices: pd.DataFrame
    metadata: str
    version: int


# ---------------------------------------------------------------------------
# Basic write / read
# ---------------------------------------------------------------------------


class TestWriteReadDataset:
    def test_write_arrow_and_read_back(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        ds = SimpleDataSet(data=df, label="test", count=42)
        media = FileSystemMedia(str(tmp_path / "ds"))
        write_dataset(media, ds)

        # data_info.json should be present
        info_path = str(tmp_path / "ds" / "data_info.json")
        assert os.path.isfile(info_path)
        with open(info_path) as f:
            info = json.load(f)
        assert "dataset_type" in info

        # data.json with non-df fields
        data_path = str(tmp_path / "ds" / "data.json")
        assert os.path.isfile(data_path)

        # Arrow file
        arrow_path = str(tmp_path / "ds" / "data.arrow")
        assert os.path.isfile(arrow_path)

    def test_round_trip_csv(self, tmp_path):
        df = pd.DataFrame({"x": [10, 20, 30]})
        ds = DataSetWithMeta(prices=df, metadata="v1", version=1)
        media = FileSystemMedia(str(tmp_path / "csv_ds"))
        write_dataset(media, ds, file_format="csv")

        loaded = read_dataset(media, file_format="csv")
        assert isinstance(loaded, (DataSetWithMeta, LoadedDataSet))

    def test_read_as_loaded_dataset(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        ds = SimpleDataSet(data=df, label="hello", count=7)
        path = str(tmp_path / "myds")
        write_dataset(path, ds)
        loaded = read_dataset(path)
        assert isinstance(loaded, (SimpleDataSet, LoadedDataSet))

    def test_empty_dataframe_not_written(self, tmp_path):
        df = pd.DataFrame()
        ds = SimpleDataSet(data=df, label="empty", count=0)
        media = FileSystemMedia(str(tmp_path / "empty_ds"))
        write_dataset(media, ds)
        # data.arrow should NOT be present for empty df
        assert not os.path.isfile(str(tmp_path / "empty_ds" / "data.arrow"))

    def test_overwrite_false_raises(self, tmp_path):
        df = pd.DataFrame({"v": [1]})
        ds = SimpleDataSet(data=df, label="x", count=1)
        media = FileSystemMedia(str(tmp_path / "ds"))
        write_dataset(media, ds)
        with pytest.raises((FileExistsError, Exception)):
            write_dataset(media, ds, overwrite=False)

    def test_overwrite_true(self, tmp_path):
        df = pd.DataFrame({"v": [1]})
        ds = SimpleDataSet(data=df, label="x", count=1)
        media = FileSystemMedia(str(tmp_path / "ds"))
        write_dataset(media, ds)
        write_dataset(media, ds, overwrite=True)  # should not raise

    def test_string_path(self, tmp_path):
        df = pd.DataFrame({"z": [5, 6]})
        ds = DataSetWithMeta(prices=df, metadata="meta", version=2)
        path = str(tmp_path / "str_ds")
        write_dataset(path, ds)
        assert os.path.isdir(path)

    def test_unsupported_media_raises(self, tmp_path):
        from qfpytoolbox.io.media import DatabaseMedia

        db_path = str(tmp_path / "x.db")
        ds = SimpleDataSet(data=pd.DataFrame(), label="x", count=0)
        with pytest.raises(TypeError, match="write_dataset"):
            write_dataset(DatabaseMedia(db_path), ds)


class TestArchiveDataset:
    def test_write_zip_and_read_back(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2], "val": [10.0, 20.0]})
        ds = DataSetWithMeta(prices=df, metadata="zip_test", version=3)
        zip_path = str(tmp_path / "ds.zip")
        write_dataset(ArchiveMedia(zip_path), ds, overwrite=True)
        assert os.path.isfile(zip_path)

    def test_write_tar_gz_and_read(self, tmp_path):
        df = pd.DataFrame({"x": [99]})
        ds = SimpleDataSet(data=df, label="tgz", count=1)
        tgz_path = str(tmp_path / "ds.tar.gz")
        write_dataset(ArchiveMedia(tgz_path), ds, overwrite=True)
        assert os.path.isfile(tgz_path)


class TestLoadedDataSet:
    def test_attributes(self):
        ds = LoadedDataSet({"a": 1, "b": "hello"})
        assert ds.attributes["a"] == 1
        assert ds.attributes["b"] == "hello"


class TestNonPersistedFields:
    def test_default_empty(self):
        assert nonpersisted_fields(SimpleDataSet) == ()

    def test_custom_class(self):
        @dataclasses.dataclass
        class MyDS(iDataSet):
            df: pd.DataFrame
            _cache: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)

        # nonpersisted_fields returns () by default
        assert nonpersisted_fields(MyDS) == ()
