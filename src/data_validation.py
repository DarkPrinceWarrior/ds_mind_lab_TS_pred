"""Data validation module using Pandera for schema validation and quality checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report containing data quality metrics and issues."""
    
    total_rows: int
    total_wells: int
    missing_values: Dict[str, int]
    duplicate_rows: int
    date_range: tuple[pd.Timestamp, pd.Timestamp]
    type_distribution: Dict[str, int]
    negative_values: Dict[str, int]
    outliers: Dict[str, int]
    issues: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            "total_rows": self.total_rows,
            "total_wells": self.total_wells,
            "missing_values": self.missing_values,
            "duplicate_rows": self.duplicate_rows,
            "date_range": [str(self.date_range[0]), str(self.date_range[1])],
            "type_distribution": self.type_distribution,
            "negative_values": self.negative_values,
            "outliers": self.outliers,
            "issues": self.issues,
            "warnings": self.warnings,
        }


class WellDataValidator:
    """Validator for well production and injection data."""
    
    def __init__(
        self,
        max_rate_percentile: float = 99.9,
        max_pressure_bar: float = 500.0,
        min_pressure_bar: float = 0.0,
    ):
        self.max_rate_percentile = max_rate_percentile
        self.max_pressure_bar = max_pressure_bar
        self.min_pressure_bar = min_pressure_bar
        
        # Schema for raw well data
        self.raw_schema = DataFrameSchema(
            {
                "date": Column(pd.Timestamp, nullable=False),
                "well": Column(str, nullable=False),
                "type": Column(str, Check.isin(["PROD", "INJ"]), nullable=False),
                "wlpt": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
                "wlpr": Column(float, nullable=True),
                "womt": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
                "womr": Column(float, nullable=True),
                "wwir": Column(float, nullable=True),
                "wwit": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
                "wthp": Column(float, nullable=True),
                "wbhp": Column(float, nullable=True),
            },
            strict=False,
            coerce=True,
        )
    
    def validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate dataframe against schema."""
        try:
            validated = self.raw_schema.validate(df, lazy=True)
            logger.info("Schema validation passed")
            return validated
        except pa.errors.SchemaErrors as exc:
            logger.error("Schema validation failed: %d errors", len(exc.failure_cases))
            for _, row in exc.failure_cases.head(10).iterrows():
                logger.error("  %s: %s", row["column"], row["check"])
            raise ValueError(f"Data validation failed with {len(exc.failure_cases)} errors") from exc
    
    def check_data_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """Generate comprehensive data quality report."""
        issues = []
        warnings = []
        
        # Basic stats
        total_rows = len(df)
        total_wells = df["well"].nunique()
        
        # Missing values
        missing_values = df.isnull().sum().to_dict()
        missing_pct = {k: v / total_rows * 100 for k, v in missing_values.items() if v > 0}
        if any(pct > 50 for pct in missing_pct.values()):
            issues.append(f"High missing values detected: {missing_pct}")
        
        # Duplicates
        duplicate_rows = df.duplicated(subset=["well", "date"]).sum()
        if duplicate_rows > 0:
            issues.append(f"Found {duplicate_rows} duplicate (well, date) pairs")
        
        # Date range
        date_range = (df["date"].min(), df["date"].max())
        
        # Type distribution
        type_dist = df["type"].value_counts().to_dict()
        
        # Negative values in rate columns
        rate_cols = ["wlpr", "womr", "wwir"]
        negative_values = {}
        for col in rate_cols:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    negative_values[col] = neg_count
                    warnings.append(f"Found {neg_count} negative values in {col}")
        
        # Outliers detection using IQR method
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ["well"]:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outlier_count > 0:
                outliers[col] = outlier_count
        
        # Check for monotonicity in cumulative columns
        for cumulative_col in ["wlpt", "womt", "wwit"]:
            if cumulative_col in df.columns:
                non_monotonic = 0
                for well in df["well"].unique():
                    well_data = df[df["well"] == well][cumulative_col].dropna()
                    if len(well_data) > 1:
                        if not well_data.is_monotonic_increasing:
                            non_monotonic += 1
                if non_monotonic > 0:
                    warnings.append(
                        f"{cumulative_col}: {non_monotonic} wells have non-monotonic cumulative values"
                    )
        
        # Check for gaps in time series
        gaps_detected = 0
        for well in df["well"].unique():
            well_dates = df[df["well"] == well]["date"].sort_values()
            if len(well_dates) > 1:
                date_diffs = well_dates.diff().dropna()
                expected_diff = pd.Timedelta(days=30)  # Monthly data
                large_gaps = (date_diffs > expected_diff * 2).sum()
                if large_gaps > 0:
                    gaps_detected += 1
        if gaps_detected > 0:
            warnings.append(f"{gaps_detected} wells have large time gaps in data")
        
        return DataQualityReport(
            total_rows=total_rows,
            total_wells=total_wells,
            missing_values={k: v for k, v in missing_values.items() if v > 0},
            duplicate_rows=duplicate_rows,
            date_range=date_range,
            type_distribution=type_dist,
            negative_values=negative_values,
            outliers=outliers,
            issues=issues,
            warnings=warnings,
        )
    
    def validate_coordinates(self, coords_df: pd.DataFrame) -> bool:
        """Validate coordinate data."""
        required_cols = {"well", "x", "y", "z"}
        if not required_cols.issubset(coords_df.columns):
            missing = required_cols - set(coords_df.columns)
            raise ValueError(f"Coordinates missing required columns: {missing}")
        
        # Check for missing coordinates
        missing_coords = coords_df[["x", "y", "z"]].isnull().any(axis=1).sum()
        if missing_coords > 0:
            logger.warning("Found %d wells with missing coordinates", missing_coords)
        
        # Check for duplicate wells
        duplicates = coords_df["well"].duplicated().sum()
        if duplicates > 0:
            raise ValueError(f"Found {duplicates} duplicate wells in coordinates")
        
        logger.info("Coordinate validation passed for %d wells", len(coords_df))
        return True


def validate_and_report(
    df: pd.DataFrame,
    coords: Optional[pd.DataFrame] = None,
    save_report: bool = True,
    output_path: Optional[str] = None,
) -> DataQualityReport:
    """Validate data and generate quality report.
    
    Args:
        df: Well data dataframe
        coords: Optional coordinates dataframe
        save_report: Whether to save report to JSON
        output_path: Path to save report
        
    Returns:
        DataQualityReport with validation results
    """
    validator = WellDataValidator()
    
    # Validate schema
    try:
        df = validator.validate_schema(df)
    except ValueError as exc:
        logger.error("Schema validation failed: %s", exc)
        raise
    
    # Generate quality report
    report = validator.check_data_quality(df)
    
    # Log summary
    logger.info("Data Quality Summary:")
    logger.info("  Total rows: %d", report.total_rows)
    logger.info("  Total wells: %d", report.total_wells)
    logger.info("  Date range: %s to %s", report.date_range[0], report.date_range[1])
    logger.info("  Type distribution: %s", report.type_distribution)
    
    if report.issues:
        logger.error("Data quality issues found:")
        for issue in report.issues:
            logger.error("  - %s", issue)
    
    if report.warnings:
        logger.warning("Data quality warnings:")
        for warning in report.warnings:
            logger.warning("  - %s", warning)
    
    # Validate coordinates if provided
    if coords is not None:
        validator.validate_coordinates(coords)
    
    # Save report
    if save_report and output_path:
        import json
        from pathlib import Path
        
        report_path = Path(output_path) / "data_quality_report.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Data quality report saved to %s", report_path)
    
    return report
