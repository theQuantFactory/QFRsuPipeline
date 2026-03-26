"""Tests for DataFrame I/O helpers."""

from __future__ import annotations

import io
import os

import pandas as pd
import pytest

from qfpytoolbox.io.dataframes import (
    read_arrow_to_df,
    read_csv_to_df,
    read_dataframe,
    write_dataframe,
)
from qfpytoolbox.io.media import ConsoleMedia, DatabaseMedia, FileSystemMedia


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]})


# ---------------------------------------------------------------------------
# read_csv_to_df / read_arrow_to_df
# ---------------------------------------------------------------------------


class TestReadCsvToDF:
    def test_reads_csv(self, sample_df, tmp_path):
        p = str(tmp_path / "data.csv")
        sample_df.to_csv(p, index=False)
        result = read_csv_to_df(p)
        pd.testing.assert_frame_equal(result, sample_df)


class TestReadArrowToDF:
    def test_reads_arrow(self, sample_df, tmp_path):
        import pyarrow.feather as feather

        p = str(tmp_path / "data.arrow")
        feather.write_feather(sample_df, p)
        result = read_arrow_to_df(p)
        pd.testing.assert_frame_equal(result, sample_df)


# ---------------------------------------------------------------------------
# read_dataframe — various src types
# ---------------------------------------------------------------------------


class TestReadDataframe:
    def test_passthrough_dataframe(self, sample_df):
        result = read_dataframe(sample_df)
        assert result is sample_df

    def test_csv_by_extension(self, sample_df, tmp_path):
        p = str(tmp_path / "data.csv")
        sample_df.to_csv(p, index=False)
        result = read_dataframe(p)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_arrow_by_extension(self, sample_df, tmp_path):
        import pyarrow.feather as feather

        p = str(tmp_path / "data.arrow")
        feather.write_feather(sample_df, p)
        result = read_dataframe(p)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_gzipped_csv(self, sample_df, tmp_path):
        p = str(tmp_path / "data.csv.gz")
        sample_df.to_csv(p, index=False, compression="gzip")
        result = read_dataframe(p)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_excel_by_extension(self, sample_df, tmp_path):
        p = str(tmp_path / "data.xlsx")
        sample_df.to_excel(p, index=False)
        result = read_dataframe(p)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_io_csv(self, sample_df):
        buf = io.StringIO()
        sample_df.to_csv(buf, index=False)
        buf.seek(0)
        result = read_dataframe(buf, format="csv")
        pd.testing.assert_frame_equal(result, sample_df)

    def test_filesystem_media(self, sample_df, tmp_path):
        p = str(tmp_path / "data.csv")
        sample_df.to_csv(p, index=False)
        media = FileSystemMedia(str(tmp_path))
        result = read_dataframe(media, filename="data.csv")
        pd.testing.assert_frame_equal(result, sample_df)

    def test_filesystem_media_direct_file(self, sample_df, tmp_path):
        p = str(tmp_path / "data.csv")
        sample_df.to_csv(p, index=False)
        media = FileSystemMedia(p)
        result = read_dataframe(media)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_database_media_table(self, sample_df, tmp_path):
        db_path = str(tmp_path / "test.db")
        media = DatabaseMedia(db_path)
        sample_df.to_sql("mytable", media.connection, if_exists="replace", index=False)
        result = read_dataframe(media, table="mytable")
        pd.testing.assert_frame_equal(result.reset_index(drop=True), sample_df.reset_index(drop=True))
        media.close()

    def test_database_media_query(self, sample_df, tmp_path):
        db_path = str(tmp_path / "test.db")
        media = DatabaseMedia(db_path)
        sample_df.to_sql("t2", media.connection, if_exists="replace", index=False)
        result = read_dataframe(media, query="SELECT a, b FROM t2")
        assert list(result.columns) == ["a", "b"]
        media.close()

    def test_database_media_no_args_raises(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        media = DatabaseMedia(db_path)
        with pytest.raises(ValueError, match="provide either"):
            read_dataframe(media)
        media.close()

    def test_console_media_raises(self):
        with pytest.raises(TypeError, match="write-only"):
            read_dataframe(ConsoleMedia())

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            read_dataframe(12345)


# ---------------------------------------------------------------------------
# write_dataframe
# ---------------------------------------------------------------------------


class TestWriteDataframe:
    def test_write_csv(self, sample_df, tmp_path):
        p = str(tmp_path / "out.csv")
        write_dataframe(p, sample_df)
        result = pd.read_csv(p)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_write_arrow(self, sample_df, tmp_path):
        import pyarrow.feather as feather

        p = str(tmp_path / "out.arrow")
        write_dataframe(p, sample_df)
        result = feather.read_feather(p)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_write_gzipped_csv(self, sample_df, tmp_path):
        p = str(tmp_path / "out.csv.gz")
        write_dataframe(p, sample_df)
        result = pd.read_csv(p, compression="gzip")
        pd.testing.assert_frame_equal(result, sample_df)

    def test_write_excel(self, sample_df, tmp_path):
        p = str(tmp_path / "out.xlsx")
        write_dataframe(p, sample_df)
        result = pd.read_excel(p)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_overwrite_false_raises(self, sample_df, tmp_path):
        p = str(tmp_path / "out.csv")
        write_dataframe(p, sample_df)
        with pytest.raises(FileExistsError, match="already exists"):
            write_dataframe(p, sample_df)

    def test_overwrite_true(self, sample_df, tmp_path):
        p = str(tmp_path / "out.csv")
        write_dataframe(p, sample_df)
        write_dataframe(p, sample_df, overwrite=True)  # should not raise

    def test_atomic_write(self, sample_df, tmp_path):
        p = str(tmp_path / "out.csv")
        write_dataframe(p, sample_df, atomic=True)
        assert os.path.isfile(p)

    def test_write_to_io(self, sample_df):
        buf = io.StringIO()
        write_dataframe(buf, sample_df, format="csv")
        buf.seek(0)
        result = pd.read_csv(buf)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_write_to_filesystem_media(self, sample_df, tmp_path):
        media = FileSystemMedia(str(tmp_path))
        write_dataframe(media, sample_df, format="csv", filename="out.csv", overwrite=True)
        result = pd.read_csv(str(tmp_path / "out.csv"))
        pd.testing.assert_frame_equal(result, sample_df)

    def test_write_to_database_media(self, sample_df, tmp_path):
        db_path = str(tmp_path / "test.db")
        media = DatabaseMedia(db_path)
        write_dataframe(media, sample_df, table="loaded")
        result = pd.read_sql_query("SELECT * FROM loaded", media.connection)
        pd.testing.assert_frame_equal(result, sample_df)
        media.close()

    def test_write_database_no_table_raises(self, sample_df, tmp_path):
        db_path = str(tmp_path / "test.db")
        media = DatabaseMedia(db_path)
        with pytest.raises(ValueError, match="table"):
            write_dataframe(media, sample_df)
        media.close()

    def test_write_non_dataframe_raises(self, tmp_path):
        with pytest.raises(TypeError):
            write_dataframe(str(tmp_path / "out.csv"), [1, 2, 3])
