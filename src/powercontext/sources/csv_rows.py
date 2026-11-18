# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline, key-addressed CSV records and their complete text evidence."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import SourceDefinition, SourceProjection
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


def _validate_document_id(document_id: str) -> None:
    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty without surrounding whitespace")  # noqa: TRY003


def _validate_document(document_id: str, primary_key: str) -> None:
    _validate_document_id(document_id)
    if not primary_key:
        raise ValueError("primary_key must be non-empty")  # noqa: TRY003


def _primary_key_index(columns: tuple[str, ...], primary_key: str) -> int:
    if not columns or any(not column.strip() for column in columns) or len(set(columns)) != len(columns):
        raise ValueError("CSV headers must be non-blank and unique")  # noqa: TRY003
    # Exact matching preserves meaningful whitespace in column names.
    return columns.index(primary_key)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Encode each identity component independently, with the Source ID bound."""
    _validate_document_id(document_id)
    if not primary_key_value.strip():
        raise ValueError("primary key values must not be blank")  # noqa: TRY003
    identity = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("encoded CSV row identity exceeds the Source ID limit")  # noqa: TRY003
    return identity


class CSVDocumentInput(BaseModel):
    """One complete UTF-8 CSV document to validate before capturing any rows."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """A complete record preserving original column order and string values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_record(self) -> CSVRowInput:
        _validate_document(self.document_id, self.primary_key)
        index = _primary_key_index(self.columns, self.primary_key)
        if len(self.values) != len(self.columns):
            raise ValueError("CSV record width must match the header")  # noqa: TRY003
        if self.values[index] != self.primary_key_value:
            raise ValueError("primary_key_value must match the record")  # noqa: TRY003
        csv_row_source_id(self.document_id, self.primary_key_value)
        return self


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate the entire document atomically, without trimming fields."""
    _validate_document(value.document_id, value.primary_key)
    # newline='' lets csv.reader retain CR/LF inside quoted fields.
    reader = csv.reader(io.StringIO(value.content.decode("utf-8-sig"), newline=""), strict=True)
    try:
        columns = tuple(next(reader, ()))
        index = _primary_key_index(columns, value.primary_key)
        rows = []
        seen: set[str] = set()
        for record in reader:
            if not record:  # Only a completely empty physical record is ignored.
                continue
            if len(record) != len(columns):
                raise ValueError("CSV record width must match the header")  # noqa: TRY003
            key = record[index]
            if key in seen:
                raise ValueError("CSV primary key values must be unique within the document")  # noqa: TRY003
            rows.append(
                CSVRowInput(
                    document_id=value.document_id,
                    primary_key=value.primary_key,
                    primary_key_value=key,
                    columns=columns,
                    values=tuple(record),
                )
            )
            seen.add(key)
    except csv.Error as error:
        raise ValueError("invalid CSV format") from error  # noqa: TRY003
    return tuple(rows)


class CSVRowSource(Source):
    """Captured original record, identified by document and primary key."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class _CSVRowTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type="csv-row",
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


class _CSVRowSourceDefinition:
    input_class = CSVRowInput
    name = "csv-row"
    version = "1"
    source_class = CSVRowSource
    projections: tuple[SourceProjection[CSVRowSource], ...] = (_CSVRowTextEvidenceProjection(),)

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        identity = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            **value.model_dump(),
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput.model_validate(source.model_dump())


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = _CSVRowSourceDefinition()
