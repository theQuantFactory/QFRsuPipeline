"""Tests for media types."""

from __future__ import annotations

import sqlite3
import sys

import pytest

from qfpytoolbox.io.media import (
    ArchiveMedia,
    ConsoleMedia,
    DatabaseMedia,
    FileSystemMedia,
    SQLDumpMedia,
    _validate_sql_identifier,
)


class TestFileSystemMedia:
    def test_basic(self):
        m = FileSystemMedia("/tmp/data")
        assert m.path == "/tmp/data"
        assert m.format == "auto"

    def test_custom_format(self):
        m = FileSystemMedia("/tmp/data", format="csv")
        assert m.format == "csv"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            FileSystemMedia("/tmp/data", format="parquet")

    def test_repr(self):
        m = FileSystemMedia("/tmp/data")
        assert "FileSystemMedia" in repr(m)


class TestDatabaseMedia:
    def test_from_string_path(self, tmp_path):
        db = str(tmp_path / "test.db")
        m = DatabaseMedia(db)
        assert m.locator == db
        assert isinstance(m.connection, sqlite3.Connection)
        m.close()

    def test_from_connection(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "x.db"))
        m = DatabaseMedia(conn)
        assert m.locator is None
        assert m.connection is conn

    def test_sqlite_url(self, tmp_path):
        db = str(tmp_path / "test.db")
        url = f"sqlite:///{db}"
        m = DatabaseMedia(url)
        m.close()

    def test_unsupported_db_type_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported db_type"):
            DatabaseMedia(str(tmp_path / "x.db"), db_type="postgres")

    def test_parameters_locator(self, tmp_path):
        db = str(tmp_path / "test.db")
        m = DatabaseMedia(db, parameters_locator="/params")
        assert m.parameters_locator == "/params"
        m.close()

    def test_context_manager(self, tmp_path):
        db = str(tmp_path / "cm.db")
        with DatabaseMedia(db) as m:
            assert m.connection is not None

    def test_repr(self, tmp_path):
        db = str(tmp_path / "test.db")
        m = DatabaseMedia(db)
        assert "DatabaseMedia" in repr(m)
        m.close()


class TestSQLDumpMedia:
    def test_basic(self, tmp_path):
        dump = tmp_path / "dump.sql"
        dump.write_text("CREATE TABLE t (id INTEGER);")
        m = SQLDumpMedia(str(dump))
        assert m.db_type == "mysql"
        assert m.in_memory

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(ValueError, match="SQL dump file not found"):
            SQLDumpMedia(str(tmp_path / "nonexistent.sql"))

    def test_unsupported_db_type_raises(self, tmp_path):
        dump = tmp_path / "dump.sql"
        dump.write_text("")
        with pytest.raises(ValueError, match="Unsupported db_type"):
            SQLDumpMedia(str(dump), db_type="oracle")

    def test_repr(self, tmp_path):
        dump = tmp_path / "dump.sql"
        dump.write_text("")
        m = SQLDumpMedia(str(dump))
        assert "SQLDumpMedia" in repr(m)


class TestConsoleMedia:
    def test_default_stdout(self):
        m = ConsoleMedia()
        assert m.stream is sys.stdout

    def test_custom_stream(self):
        import io

        buf = io.StringIO()
        m = ConsoleMedia(buf)
        assert m.stream is buf

    def test_repr(self):
        m = ConsoleMedia()
        assert "ConsoleMedia" in repr(m)


class TestArchiveMedia:
    def test_zip_inferred(self):
        m = ArchiveMedia("/tmp/out.zip")
        assert m.format == "zip"

    def test_tar_gz_inferred(self):
        m = ArchiveMedia("/tmp/out.tar.gz")
        assert m.format == "tar_gz"

    def test_tgz_inferred(self):
        m = ArchiveMedia("/tmp/out.tgz")
        assert m.format == "tar_gz"

    def test_explicit_format(self):
        m = ArchiveMedia("/tmp/out.bin", format="zip")
        assert m.format == "zip"

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Cannot determine archive format"):
            ArchiveMedia("/tmp/out.bin")

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported archive format"):
            ArchiveMedia("/tmp/out.zip", format="rar")

    def test_repr(self):
        m = ArchiveMedia("/tmp/out.zip")
        assert "ArchiveMedia" in repr(m)


class TestValidateSqlIdentifier:
    def test_valid(self):
        assert _validate_sql_identifier("my_table") == "my_table"

    def test_reserved_raises(self):
        with pytest.raises(ValueError, match="reserved SQL keyword"):
            _validate_sql_identifier("SELECT")

    def test_starts_with_digit_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            _validate_sql_identifier("1table")

    def test_special_char_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            _validate_sql_identifier("my-table")
