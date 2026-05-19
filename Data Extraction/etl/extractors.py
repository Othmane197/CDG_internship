"""Data extraction utilities for ETL pipelines."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


class ExtractorError(RuntimeError):
    """Raised when a data extraction operation fails."""


def read_flat_file(
    path: str,
    *,
    file_type: Optional[str] = None,
    encoding: str = "utf-8",
    sep: str = ",",
    sheet_name: Optional[str | int] = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read CSV or Excel files into a pandas DataFrame.

    Args:
        path: Path to the input file.
        file_type: Optional explicit file type ("csv" or "excel").
        encoding: File encoding for CSV input.
        sep: Delimiter for CSV input.
        sheet_name: Excel sheet name or index to read.
        **kwargs: Extra keyword arguments passed to pandas readers.

    Returns:
        A pandas DataFrame containing the file contents.
    """
    try:
        detected_type = file_type
        if detected_type is None:
            ext = os.path.splitext(path)[1].lower()
            if ext in {".csv", ".txt"}:
                detected_type = "csv"
            elif ext in {".xls", ".xlsx", ".xlsm"}:
                detected_type = "excel"

        if detected_type == "csv":
            return pd.read_csv(path, encoding=encoding, sep=sep, **kwargs)
        if detected_type == "excel":
            return pd.read_excel(path, sheet_name=sheet_name, **kwargs)

        raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")
    except Exception as exc:
        logger.exception("Flat file extraction failed: %s", path)
        raise ExtractorError(f"Flat file extraction failed for {path}.") from exc


def extract_pdf_table(
    pdf_path: str,
    *,
    page_number: int = 1,
    table_settings: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Extract a table from a native PDF page into a DataFrame.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-based page number to extract from.
        table_settings: Optional pdfplumber table extraction settings.

    Returns:
        A cleaned DataFrame built from the extracted table.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise ExtractorError("pdfplumber is required for PDF extraction.") from exc

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                raise ValueError("Page number out of range.")

            page = pdf.pages[page_number - 1]
            if table_settings:
                table = page.extract_table(table_settings)
            else:
                table = page.extract_table()

            if not table:
                raise ValueError("No table found on the requested page.")

            header, *rows = table
            df = pd.DataFrame(rows, columns=header)
            return _clean_dataframe(df)
    except Exception as exc:
        logger.exception("PDF table extraction failed: %s", pdf_path)
        raise ExtractorError(f"PDF table extraction failed for {pdf_path}.") from exc


def extract_text_from_scanned_image(
    image_path: str,
    *,
    lang: str = "fra+eng",
    psm: int = 6,
    oem: int = 3,
) -> str:
    """Extract raw text from a scanned document image using OCR.

    Args:
        image_path: Path to the input image.
        lang: Tesseract language configuration string.
        psm: Tesseract page segmentation mode.
        oem: Tesseract OCR engine mode.

    Returns:
        The extracted raw text.
    """
    try:
        import cv2
        import pytesseract
    except ImportError as exc:
        raise ExtractorError("opencv-python and pytesseract are required for OCR.") from exc

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to read the image file.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        thresholded = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]

        config = f"--oem {oem} --psm {psm}"
        return pytesseract.image_to_string(thresholded, lang=lang, config=config)
    except Exception as exc:
        logger.exception("OCR extraction failed: %s", image_path)
        raise ExtractorError(f"OCR extraction failed for {image_path}.") from exc


def query_database_to_df(
    connection_uri: str,
    sql: str,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Execute a SQL query safely and return the results as a DataFrame.

    Args:
        connection_uri: SQLAlchemy database URI.
        sql: SQL query to execute.
        params: Optional parameters for the query.

    Returns:
        A pandas DataFrame containing the query results.
    """
    try:
        from sqlalchemy import create_engine, text as sql_text
    except ImportError as exc:
        raise ExtractorError("sqlalchemy is required for database extraction.") from exc

    engine = create_engine(connection_uri)
    try:
        with engine.connect() as connection:
            return pd.read_sql_query(sql_text(sql), connection, params=params)
    except Exception as exc:
        logger.exception("Database query failed.")
        raise ExtractorError("Database query failed.") from exc
    finally:
        engine.dispose()


def fetch_api_json_to_df(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    json_path: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Fetch JSON from a web API and flatten it into a DataFrame.

    Args:
        url: API endpoint URL.
        params: Query parameters for the request.
        headers: Optional request headers.
        timeout: Request timeout in seconds.
        json_path: Optional iterable of keys to reach the record list.

    Returns:
        A pandas DataFrame containing the flattened JSON response.
    """
    try:
        import requests
    except ImportError as exc:
        raise ExtractorError("requests is required for API extraction.") from exc

    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()

        payload = response.json()
        data = _extract_json_path(payload, json_path)
        records = _normalize_records(data)
        return pd.json_normalize(records)
    except Exception as exc:
        logger.exception("API extraction failed: %s", url)
        raise ExtractorError(f"API extraction failed for {url}.") from exc


def _extract_json_path(payload: Any, json_path: Optional[Iterable[str]]) -> Any:
    if not json_path:
        return payload

    current = payload
    for key in json_path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            raise ValueError("json_path does not exist in payload.")
    return current


def _normalize_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        if data and not isinstance(data[0], dict):
            return [{"value": item} for item in data]
        return data

    if isinstance(data, dict):
        return [data]

    raise ValueError("API payload is not a list or dict.")


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    cleaned = cleaned.applymap(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return cleaned.reset_index(drop=True)


__all__ = [
    "ExtractorError",
    "read_flat_file",
    "extract_pdf_table",
    "extract_text_from_scanned_image",
    "query_database_to_df",
    "fetch_api_json_to_df",
]
