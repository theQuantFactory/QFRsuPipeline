"""Media interface — abstract base and concrete backends.

Mirrors Julia's ``iSourceMedia`` hierarchy from
``MyJuliaToolbox.jl/src/io/media.jl``.
"""

from __future__ import annotations

import re
import sys
from abc import ABC
from typing import Any

__all__ = [
    "iSourceMedia",
    "FileSystemMedia",
    "DatabaseMedia",
    "SQLDumpMedia",
    "ConsoleMedia",
    "ArchiveMedia",
]

# ---------------------------------------------------------------------------
# SQLite reserved keywords — must not be used as bare (unquoted) identifiers.
# Source: https://www.sqlite.org/lang_keywords.html
# ---------------------------------------------------------------------------
_SQL_RESERVED_KEYWORDS: frozenset[str] = frozenset(
    {
        "ABORT",
        "ACTION",
        "ADD",
        "AFTER",
        "ALL",
        "ALTER",
        "ANALYZE",
        "AND",
        "AS",
        "ASC",
        "ATTACH",
        "AUTOINCREMENT",
        "BEFORE",
        "BEGIN",
        "BETWEEN",
        "BY",
        "CASCADE",
        "CASE",
        "CAST",
        "CHECK",
        "COLLATE",
        "COLUMN",
        "COMMIT",
        "CONFLICT",
        "CONSTRAINT",
        "CREATE",
        "CROSS",
        "CURRENT_DATE",
        "CURRENT_TIME",
        "CURRENT_TIMESTAMP",
        "DATABASE",
        "DEFAULT",
        "DEFERRABLE",
        "DEFERRED",
        "DELETE",
        "DESC",
        "DETACH",
        "DISTINCT",
        "DROP",
        "EACH",
        "ELSE",
        "END",
        "ESCAPE",
        "EXCEPT",
        "EXCLUSIVE",
        "EXISTS",
        "EXPLAIN",
        "FAIL",
        "FOR",
        "FOREIGN",
        "FROM",
        "FULL",
        "GLOB",
        "GROUP",
        "HAVING",
        "IF",
        "IGNORE",
        "IMMEDIATE",
        "IN",
        "INDEX",
        "INDEXED",
        "INITIALLY",
        "INNER",
        "INSERT",
        "INSTEAD",
        "INTERSECT",
        "INTO",
        "IS",
        "ISNULL",
        "JOIN",
        "KEY",
        "LEFT",
        "LIKE",
        "LIMIT",
        "MATCH",
        "NATURAL",
        "NO",
        "NOT",
        "NOTNULL",
        "NULL",
        "OF",
        "OFFSET",
        "ON",
        "OR",
        "ORDER",
        "OUTER",
        "PLAN",
        "PRAGMA",
        "PRIMARY",
        "QUERY",
        "RAISE",
        "RECURSIVE",
        "REFERENCES",
        "REGEXP",
        "REINDEX",
        "RELEASE",
        "RENAME",
        "REPLACE",
        "RESTRICT",
        "RIGHT",
        "ROLLBACK",
        "ROW",
        "SAVEPOINT",
        "SELECT",
        "SET",
        "TABLE",
        "TEMP",
        "TEMPORARY",
        "THEN",
        "TO",
        "TRANSACTION",
        "TRIGGER",
        "UNION",
        "UNIQUE",
        "UPDATE",
        "USING",
        "VACUUM",
        "VALUES",
        "VIEW",
        "VIRTUAL",
        "WHEN",
        "WHERE",
        "WITH",
        "WITHOUT",
    }
)


def _validate_sql_identifier(identifier: str, field: str = "identifier") -> str:
    """Validate ``identifier`` as a safe bare SQL name.  Returns it unchanged."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", identifier):
        raise ValueError(f"Invalid {field} '{identifier}'. Use only letters, digits, and underscore.")
    if identifier.upper() in _SQL_RESERVED_KEYWORDS:
        raise ValueError(f"Invalid {field} '{identifier}': reserved SQL keyword. Choose a different name.")
    return identifier


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class iSourceMedia(ABC):  # noqa: B024
    """Abstract base type for all media backends.

    Subclasses represent a specific storage or transport mechanism (file system,
    database, SQL dump, console, …).  All concrete I/O methods are optional and
    are added by the appropriate modules (``dataframes``, ``logger``, etc.).
    """


# ---------------------------------------------------------------------------
# FileSystemMedia
# ---------------------------------------------------------------------------


class FileSystemMedia(iSourceMedia):
    """A media backend for file-system paths.

    Parameters
    ----------
    path:
        Path to the file or directory.
    format:
        Default format for I/O operations (``"csv"``, ``"arrow"``, ``"auto"``).
        ``"auto"`` (default) infers the format from the file extension.
    """

    def __init__(self, path: str, *, format: str = "auto") -> None:
        _VALID_FORMATS = {"auto", "csv", "arrow", "excel"}
        if format not in _VALID_FORMATS:
            raise ValueError(f"Unsupported format: {format!r}. Choose from: {sorted(_VALID_FORMATS)}")
        self.path = path
        self.format = format

    def __repr__(self) -> str:
        return f"FileSystemMedia({self.path!r}, format={self.format!r})"


# ---------------------------------------------------------------------------
# DatabaseMedia
# ---------------------------------------------------------------------------


def _sqlite_path_from_locator(locator: str) -> str:
    if locator.startswith("sqlite:///"):
        return locator[10:]
    if locator.startswith("sqlite://"):
        return locator[9:]
    return locator


class DatabaseMedia(iSourceMedia):
    """A media backend for database sources (currently SQLite only).

    Parameters
    ----------
    locator:
        Database path, URL (``"sqlite:///path/to/db"``), or a pre-opened
        ``sqlite3.Connection``.
    db_type:
        Database type.  Only ``"sqlite"`` is currently supported.
    parameters_locator:
        Optional path root used by :func:`~qfpytoolbox.parameters.read_parameters`.
    """

    def __init__(
        self,
        locator: Any,
        *,
        db_type: str = "sqlite",
        parameters_locator: str | None = None,
    ) -> None:
        import sqlite3

        if db_type != "sqlite":
            raise ValueError(f"Unsupported db_type: {db_type!r}. Currently supported: 'sqlite'")

        self.db_type = db_type
        self.parameters_locator = parameters_locator

        if isinstance(locator, str):
            self.locator: str | None = locator
            db_path = _sqlite_path_from_locator(locator)
            self.connection: Any = sqlite3.connect(db_path, check_same_thread=False)
            self._owns_connection = True
        elif isinstance(locator, sqlite3.Connection):
            self.locator = None
            self.connection = locator
            self._owns_connection = False
        else:
            raise TypeError(f"locator must be a str path/URL or sqlite3.Connection, got {type(locator).__name__}")

    def __repr__(self) -> str:
        loc = self.locator or "<connection>"
        return f"DatabaseMedia({loc!r}, db_type={self.db_type!r})"

    def close(self) -> None:
        """Close the connection if this object owns it."""
        if self._owns_connection and self.connection is not None:
            self.connection.close()

    def __enter__(self) -> DatabaseMedia:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# SQLDumpMedia
# ---------------------------------------------------------------------------


class SQLDumpMedia(iSourceMedia):
    """A read-only media backend that loads a SQL dump into an in-memory SQLite DB.

    Parameters
    ----------
    path:
        Path to the ``.sql`` dump file.
    db_type:
        Type of source database dump (``"mysql"`` or ``"postgresql"``).
    in_memory:
        When ``True`` (default) the SQLite DB lives in memory.
    parameters_locator:
        Optional path root used by :func:`~qfpytoolbox.parameters.read_parameters`.
    """

    def __init__(
        self,
        path: str,
        *,
        db_type: str = "mysql",
        in_memory: bool = True,
        parameters_locator: str | None = None,
    ) -> None:
        import os

        if not os.path.isfile(path):
            raise ValueError(f"SQL dump file not found: {path}")
        if db_type not in ("mysql", "postgresql"):
            raise ValueError(f"Unsupported db_type: {db_type!r}. Must be 'mysql' or 'postgresql'")

        self.path = path
        self.db_type = db_type
        self.in_memory = in_memory
        self.parameters_locator = parameters_locator

    def __repr__(self) -> str:
        return f"SQLDumpMedia({self.path!r}, db_type={self.db_type!r})"


# ---------------------------------------------------------------------------
# ConsoleMedia
# ---------------------------------------------------------------------------


class ConsoleMedia(iSourceMedia):
    """A write-only media backend that outputs to a stream (default: ``sys.stdout``).

    Parameters
    ----------
    stream:
        Output stream.  Defaults to ``sys.stdout``.
    """

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream if stream is not None else sys.stdout

    def __repr__(self) -> str:
        name = getattr(self.stream, "name", repr(self.stream))
        return f"ConsoleMedia(stream={name!r})"


# ---------------------------------------------------------------------------
# ArchiveMedia
# ---------------------------------------------------------------------------


def _archive_format_from_path(path: str) -> str:
    lp = path.lower()
    if lp.endswith(".zip"):
        return "zip"
    if lp.endswith(".tar.gz") or lp.endswith(".tgz"):
        return "tar_gz"
    raise ValueError(
        f"Cannot determine archive format from path: {path!r}. "
        "Supported extensions: .zip, .tar.gz, .tgz. "
        "Use ArchiveMedia(path, format='zip') to specify explicitly."
    )


class ArchiveMedia(iSourceMedia):
    """A media backend for archive files (``.zip`` or ``.tar.gz``).

    Parameters
    ----------
    path:
        Path to the archive file (need not exist yet for writes).
    format:
        Archive format — ``"zip"`` or ``"tar_gz"``.  Inferred from extension by
        default.
    """

    def __init__(self, path: str, *, format: str | None = None) -> None:
        resolved = format if format is not None else _archive_format_from_path(path)
        if resolved not in ("zip", "tar_gz"):
            raise ValueError(f"Unsupported archive format: {resolved!r}. Supported: 'zip', 'tar_gz'")
        self.path = path
        self.format = resolved

    def __repr__(self) -> str:
        return f"ArchiveMedia({self.path!r}, format={self.format!r})"
