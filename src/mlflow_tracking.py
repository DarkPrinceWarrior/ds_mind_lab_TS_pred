"""MLflow tracking integration for experiment management."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracker for pipeline runs."""
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "wlpr_forecasting",
        run_name: Optional[str] = None,
    ):
        """Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI (None for local)
            experiment_name: Name of the experiment
            run_name: Optional name for this run
        """
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_id = None
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        logger.info("MLflow tracking initialized: %s", self.tracking_uri)
    
    def start_run(self, run_name: Optional[str] = None):
        """Start MLflow run."""
        self.run_name = run_name or self.run_name
        mlflow.start_run(run_name=self.run_name)
        self.run_id = mlflow.active_run().info.run_id
        logger.info("Started MLflow run: %s (ID: %s)", self.run_name, self.run_id)
    
    def end_run(self):
        """End MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
            logger.info("Ended MLflow run: %s", self.run_id)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        try:
            # Flatten nested dicts
            flat_params = self._flatten_dict(params)
            for key, value in flat_params.items():
                if value is not None:
                    mlflow.log_param(key, value)
            logger.debug("Logged %d parameters to MLflow", len(flat_params))
        except Exception as exc:
            logger.warning("Failed to log params: %s", exc)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        try:
            for key, value in metrics.items():
                if value is not None and isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
            logger.debug("Logged %d metrics to MLflow", len(metrics))
        except Exception as exc:
            logger.warning("Failed to log metrics: %s", exc)
    
    def log_artifact(self, local_path: Path, artifact_path: Optional[str] = None):
        """Log file artifact to MLflow."""
        try:
            mlflow.log_artifact(str(local_path), artifact_path)
            logger.debug("Logged artifact: %s", local_path)
        except Exception as exc:
            logger.warning("Failed to log artifact %s: %s", local_path, exc)
    
    def log_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        artifact_path: Optional[str] = None,
    ):
        """Log pandas DataFrame as artifact."""
        try:
            temp_path = Path(f"temp_{name}.csv")
            df.to_csv(temp_path, index=False)
            self.log_artifact(temp_path, artifact_path)
            temp_path.unlink()
        except Exception as exc:
            logger.warning("Failed to log dataframe %s: %s", name, exc)
    
    def log_dict(self, dictionary: Dict, name: str, artifact_path: Optional[str] = None):
        """Log dictionary as JSON artifact."""
        import json
        
        try:
            temp_path = Path(f"temp_{name}.json")
            with open(temp_path, "w") as f:
                json.dump(dictionary, f, indent=2, default=str)
            self.log_artifact(temp_path, artifact_path)
            temp_path.unlink()
        except Exception as exc:
            logger.warning("Failed to log dict %s: %s", name, exc)
    
    def log_config(self, config: Any):
        """Log pipeline configuration."""
        try:
            if hasattr(config, "__dict__"):
                config_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config)
            else:
                config_dict = dict(config)
            
            self.log_params(config_dict)
            self.log_dict(config_dict, "config")
        except Exception as exc:
            logger.warning("Failed to log config: %s", exc)
    
    def log_model_summary(self, model_info: Dict[str, Any]):
        """Log model architecture and training summary."""
        try:
            self.log_dict(model_info, "model_summary")
            
            # Log key model params
            if "n_parameters" in model_info:
                mlflow.log_metric("n_parameters", model_info["n_parameters"])
        except Exception as exc:
            logger.warning("Failed to log model summary: %s", exc)
    
    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the run."""
        try:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        except Exception as exc:
            logger.warning("Failed to set tags: %s", exc)
    
    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                # Convert lists to strings for MLflow
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(exc_val))
        else:
            mlflow.set_tag("status", "success")
        
        self.end_run()


def create_tracker(
    config: Any,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> Optional[MLflowTracker]:
    """Create MLflow tracker if available.
    
    Args:
        config: Pipeline configuration
        run_name: Optional run name
        tracking_uri: Optional tracking URI
    
    Returns:
        MLflowTracker instance or None if MLflow not available
    """
    try:
        from datetime import datetime
        
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        tracker = MLflowTracker(
            tracking_uri=tracking_uri,
            experiment_name="wlpr_forecasting",
            run_name=run_name,
        )
        
        return tracker
    except ImportError:
        logger.warning("MLflow not available, tracking disabled")
        return None
    except Exception as exc:
        logger.warning("Failed to initialize MLflow tracker: %s", exc)
        return None
