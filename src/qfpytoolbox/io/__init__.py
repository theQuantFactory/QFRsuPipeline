"""IO sub-package: media abstractions, DataFrame I/O, and async logging."""

from __future__ import annotations

from qfpytoolbox.io.dataframes import (
    read_arrow_to_df,
    read_csv_to_df,
    read_dataframe,
    write_dataframe,
)
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
from qfpytoolbox.io.media import (
    ArchiveMedia,
    ConsoleMedia,
    DatabaseMedia,
    FileSystemMedia,
    SQLDumpMedia,
    iSourceMedia,
)

__all__ = [
    # media
    "iSourceMedia",
    "FileSystemMedia",
    "DatabaseMedia",
    "SQLDumpMedia",
    "ConsoleMedia",
    "ArchiveMedia",
    # dataframes
    "read_dataframe",
    "write_dataframe",
    "read_csv_to_df",
    "read_arrow_to_df",
    # logger
    "LogRecord",
    "AsyncLogger",
    "log_debug",
    "log_info",
    "log_warn",
    "log_error",
    "flush_logger",
    "stop_logger",
    "dropped_logs",
]
