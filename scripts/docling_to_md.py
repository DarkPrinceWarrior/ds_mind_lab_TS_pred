#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path


SUPPORTED_EXTS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".ppt",
    ".html",
    ".htm",
    ".md",
    ".markdown",
    ".txt",
    ".rtf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
}


def iter_input_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    files = [p for p in input_path.glob(pattern) if p.is_file()]
    return files


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def build_output_path(output_dir: Path, input_root: Path | None, src: Path) -> Path:
    if input_root and input_root.is_dir():
        rel = src.relative_to(input_root)
    else:
        rel = Path(src.name)
    return (output_dir / rel).with_suffix(".md")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert documents to Markdown using Docling."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=".",
        help="Input file or directory (default: current directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="files_md",
        help="Output directory for Markdown files (default: files_md).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Scan input directory recursively.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Try to convert all files, ignoring extension filter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown files.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("docling_to_md")

    try:
        import docling
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        
        try:
            from importlib.metadata import version
            ver = version("docling")
        except Exception:
            ver = "unknown"
        logger.info(f"Using Docling version: {ver}")
    except ImportError as exc:
        print("Docling is not available. Please install the required packages:")
        print("pip install docling pandas tabulate")
        print(f"Import error: {exc}")
        return 2

    files = iter_input_files(input_path, args.recursive)
    if not files:
        logger.error(f"No files found at: {input_path}")
        return 1

    if not args.no_filter:
        files = [f for f in files if is_supported(f)]
        if not files:
            logger.error(
                "No supported files found. Use --no-filter to try all files."
            )
            return 1

    # Configure pipeline options for better formula extraction
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    # Attempt to enable formula enrichment if available in this version
    try:
        pipeline_options.do_formula_enrichment = True
        logger.info("Formula enrichment enabled.")
    except AttributeError:
        logger.warning("Formula enrichment option not found in PdfPipelineOptions.")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    ok = 0
    failed = 0

    for src in files:
        try:
            result = converter.convert(str(src))
            doc = result.document
            if doc is None:
                raise RuntimeError("Conversion returned no document.")

            md = doc.export_to_markdown()
            out_path = build_output_path(output_dir, input_path, src)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists() and not args.overwrite:
                logger.info(f"Skip (exists): {out_path}")
                continue

            out_path.write_text(md, encoding="utf-8")
            ok += 1
            logger.info(f"OK: {src} -> {out_path}")
        except Exception as exc:
            failed += 1
            logger.error(f"FAILED: {src} ({exc})")

    logger.info(
        f"Done. Success: {ok}, Failed: {failed}, Output: {output_dir}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
