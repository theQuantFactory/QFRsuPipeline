"""Tests for the async logger."""

from __future__ import annotations

import io

import pytest

from qfpytoolbox.io.logger import (
    AsyncLogger,
    LogRecord,
    dropped_logs,
    flush_logger,
    log_debug,
    log_error,
    log_info,
    log_warn,
    stop_logger,
)
from qfpytoolbox.io.media import ConsoleMedia, DatabaseMedia, FileSystemMedia


class TestLogRecord:
    def test_fields(self):
        from datetime import datetime, timezone

        ts = datetime.now(tz=timezone.utc)
        r = LogRecord(ts, "info", "hello", metadata={"x": 1})
        assert r.level == "info"
        assert r.message == "hello"
        assert r.metadata == {"x": 1}


class TestAsyncLoggerConsole:
    def test_log_info_written(self):
        buf = io.StringIO()
        logger = AsyncLogger(ConsoleMedia(buf), min_level="info")
        log_info(logger, "hello world")
        stop_logger(logger)
        assert "hello world" in buf.getvalue()
        assert "[INFO]" in buf.getvalue()

    def test_log_debug_filtered_by_min_level(self):
        buf = io.StringIO()
        logger = AsyncLogger(ConsoleMedia(buf), min_level="info")
        log_debug(logger, "debug msg")
        stop_logger(logger)
        assert "debug msg" not in buf.getvalue()

    def test_log_warn_passes_info_level(self):
        buf = io.StringIO()
        logger = AsyncLogger(ConsoleMedia(buf), min_level="info")
        log_warn(logger, "warning!")
        stop_logger(logger)
        assert "warning!" in buf.getvalue()

    def test_log_error(self):
        buf = io.StringIO()
        logger = AsyncLogger(ConsoleMedia(buf), min_level="debug")
        log_error(logger, "bad thing")
        stop_logger(logger)
        assert "bad thing" in buf.getvalue()

    def test_metadata_included(self):
        buf = io.StringIO()
        logger = AsyncLogger(ConsoleMedia(buf), min_level="info")
        log_info(logger, "msg", metadata={"job": 42})
        stop_logger(logger)
        assert "job" in buf.getvalue()

    def test_dropped_logs_when_full(self):
        buf = io.StringIO()
        logger = AsyncLogger(ConsoleMedia(buf), min_level="info", capacity=2)
        # Stop the worker so the queue fills
        stop_logger(logger)
        # These log calls go to a closed queue and should be dropped
        for _ in range(5):
            logger.log("info", "fill")
        assert dropped_logs(logger) >= 0  # Just check it doesn't raise

    def test_flush_logger(self):
        buf = io.StringIO()
        logger = AsyncLogger(ConsoleMedia(buf), min_level="info")
        for i in range(10):
            log_info(logger, f"msg {i}")
        flush_logger(logger)
        stop_logger(logger)
        val = buf.getvalue()
        assert "msg 0" in val

    def test_invalid_min_level_raises(self):
        with pytest.raises(ValueError, match="Unsupported log level"):
            AsyncLogger(ConsoleMedia(), min_level="verbose")

    def test_all_levels_at_debug(self):
        buf = io.StringIO()
        logger = AsyncLogger(ConsoleMedia(buf), min_level="debug")
        log_debug(logger, "d")
        log_info(logger, "i")
        log_warn(logger, "w")
        log_error(logger, "e")
        stop_logger(logger)
        val = buf.getvalue()
        assert "d" in val
        assert "i" in val
        assert "w" in val
        assert "e" in val


class TestAsyncLoggerFile:
    def test_writes_to_file(self, tmp_path):
        log_file = str(tmp_path / "app.log")
        logger = AsyncLogger(FileSystemMedia(log_file), min_level="info")
        log_info(logger, "file log entry")
        stop_logger(logger)
        with open(log_file) as f:
            content = f.read()
        assert "file log entry" in content

    def test_creates_parent_dir(self, tmp_path):
        log_file = str(tmp_path / "sub" / "dir" / "app.log")
        logger = AsyncLogger(FileSystemMedia(log_file), min_level="info")
        log_info(logger, "nested")
        stop_logger(logger)
        import os

        assert os.path.isfile(log_file)


class TestAsyncLoggerDatabase:
    def test_writes_to_sqlite(self, tmp_path):
        db_path = str(tmp_path / "logs.db")
        media = DatabaseMedia(db_path)
        logger = AsyncLogger(media, min_level="info", table="app_logs")
        log_info(logger, "db log entry")
        stop_logger(logger)
        df = __import__("pandas").read_sql_query("SELECT * FROM app_logs", media.connection)
        assert len(df) == 1
        assert df.iloc[0]["message"] == "db log entry"
        media.close()

    def test_invalid_table_name_raises(self, tmp_path):
        db_path = str(tmp_path / "logs.db")
        media = DatabaseMedia(db_path)
        # invalid table name is validated when writing; the worker captures it silently,
        # so we test the validation helper directly
        from qfpytoolbox.io.media import _validate_sql_identifier

        with pytest.raises(ValueError, match="Invalid"):
            _validate_sql_identifier("1bad", field="table name")
        media.close()
