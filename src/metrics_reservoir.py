"""Specialized metrics for reservoir engineering and well production forecasting."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

EPSILON = 1e-9


def decline_curve_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_idx: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Metrics specific to production decline curves.
    
    Research basis: Standard metrics in petroleum engineering
    
    Args:
        y_true: Actual production rates
        y_pred: Predicted production rates  
        time_idx: Time indices (months)
    
    Returns:
        Dictionary with decline-specific metrics
    """
    metrics = {}
    
    # 1. Peak production error
    peak_true = np.max(y_true)
    peak_pred = np.max(y_pred)
    metrics["peak_production_error_pct"] = 100 * abs(peak_pred - peak_true) / (peak_true + EPSILON)
    
    # 2. Time to peak error
    if time_idx is not None and len(time_idx) == len(y_true):
        peak_time_true = time_idx[np.argmax(y_true)]
        peak_time_pred = time_idx[np.argmax(y_pred)]
        metrics["peak_time_error_months"] = abs(peak_time_pred - peak_time_true)
    
    # 3. Decline rate error
    # Fit exponential decline: q(t) = q0 * exp(-D*t)
    if time_idx is not None and len(time_idx) == len(y_true):
        try:
            # True decline rate
            peak_idx_true = np.argmax(y_true)
            if peak_idx_true < len(y_true) - 1:
                decline_data_true = y_true[peak_idx_true:]
                decline_time_true = time_idx[peak_idx_true:] - time_idx[peak_idx_true]
                
                # Log-linear fit
                valid_mask = decline_data_true > EPSILON
                if valid_mask.sum() > 3:
                    log_q_true = np.log(decline_data_true[valid_mask] + EPSILON)
                    t_true = decline_time_true[valid_mask]
                    
                    slope_true, _ = np.polyfit(t_true, log_q_true, 1)
                    decline_rate_true = -slope_true
                    
                    # Predicted decline rate
                    peak_idx_pred = np.argmax(y_pred)
                    if peak_idx_pred < len(y_pred) - 1:
                        decline_data_pred = y_pred[peak_idx_pred:]
                        decline_time_pred = time_idx[peak_idx_pred:] - time_idx[peak_idx_pred]
                        
                        valid_mask_pred = decline_data_pred > EPSILON
                        if valid_mask_pred.sum() > 3:
                            log_q_pred = np.log(decline_data_pred[valid_mask_pred] + EPSILON)
                            t_pred = decline_time_pred[valid_mask_pred]
                            
                            slope_pred, _ = np.polyfit(t_pred, log_q_pred, 1)
                            decline_rate_pred = -slope_pred
                            
                            metrics["decline_rate_true"] = decline_rate_true
                            metrics["decline_rate_pred"] = decline_rate_pred
                            metrics["decline_rate_error_pct"] = 100 * abs(
                                decline_rate_pred - decline_rate_true
                            ) / (decline_rate_true + EPSILON)
        except Exception as e:
            logger.debug("Could not compute decline rate: %s", e)
    
    # 4. Cumulative production error
    cum_true = np.cumsum(y_true)
    cum_pred = np.cumsum(y_pred)
    metrics["cumulative_error_pct"] = 100 * abs(cum_pred[-1] - cum_true[-1]) / (cum_true[-1] + EPSILON)
    
    # 5. Plateau duration error (time above 90% of peak)
    threshold_true = 0.9 * peak_true
    threshold_pred = 0.9 * peak_pred
    
    plateau_mask_true = y_true >= threshold_true
    plateau_mask_pred = y_pred >= threshold_pred
    
    plateau_months_true = plateau_mask_true.sum()
    plateau_months_pred = plateau_mask_pred.sum()
    
    metrics["plateau_duration_true"] = plateau_months_true
    metrics["plateau_duration_pred"] = plateau_months_pred
    metrics["plateau_duration_error"] = abs(plateau_months_pred - plateau_months_true)
    
    return metrics


def reservoir_pressure_metrics(
    pressure_true: np.ndarray,
    pressure_pred: np.ndarray,
    rate_true: np.ndarray,
) -> Dict[str, float]:
    """Metrics for pressure prediction.
    
    Args:
        pressure_true: Actual pressures
        pressure_pred: Predicted pressures
        rate_true: Production rates
    
    Returns:
        Pressure-specific metrics
    """
    metrics = {}
    
    # 1. Pressure maintenance error
    initial_p_true = pressure_true[0]
    initial_p_pred = pressure_pred[0]
    
    final_p_true = pressure_true[-1]
    final_p_pred = pressure_pred[-1]
    
    drawdown_true = initial_p_true - final_p_true
    drawdown_pred = initial_p_pred - final_p_pred
    
    metrics["drawdown_error_bar"] = abs(drawdown_pred - drawdown_true)
    metrics["drawdown_error_pct"] = 100 * abs(drawdown_pred - drawdown_true) / (drawdown_true + EPSILON)
    
    # 2. Productivity index error
    # PI = rate / drawdown
    reservoir_p = 200.0  # Assumed reservoir pressure (bar)
    
    pi_true = rate_true / np.maximum(reservoir_p - pressure_true, 1.0)
    pi_pred = rate_true / np.maximum(reservoir_p - pressure_pred, 1.0)
    
    metrics["productivity_index_mae"] = np.mean(np.abs(pi_pred - pi_true))
    
    # 3. Pressure gradient error
    dp_dt_true = np.gradient(pressure_true)
    dp_dt_pred = np.gradient(pressure_pred)
    
    metrics["pressure_gradient_rmse"] = np.sqrt(np.mean((dp_dt_pred - dp_dt_true) ** 2))
    
    return metrics


def injection_efficiency_metrics(
    production_true: np.ndarray,
    production_pred: np.ndarray,
    injection_rates: np.ndarray,
    voidage_replacement: Optional[float] = None,
) -> Dict[str, float]:
    """Metrics for injection-production relationship.
    
    Args:
        production_true: Actual production
        production_pred: Predicted production
        injection_rates: Injection rates
        voidage_replacement: Target VRR (default: 1.0)
    
    Returns:
        Injection efficiency metrics
    """
    metrics = {}
    
    if voidage_replacement is None:
        voidage_replacement = 1.0
    
    # 1. Voidage Replacement Ratio (VRR)
    # VRR = cumulative injection / cumulative production
    cum_inj = np.cumsum(injection_rates)
    cum_prod_true = np.cumsum(production_true)
    cum_prod_pred = np.cumsum(production_pred)
    
    vrr_true = cum_inj[-1] / (cum_prod_true[-1] + EPSILON)
    vrr_pred = cum_inj[-1] / (cum_prod_pred[-1] + EPSILON)
    
    metrics["vrr_true"] = vrr_true
    metrics["vrr_pred"] = vrr_pred
    metrics["vrr_error"] = abs(vrr_pred - vrr_true)
    
    # 2. Injection efficiency (production increase per unit injection)
    # Compute correlation between injection and production change
    prod_change_true = np.diff(production_true, prepend=production_true[0])
    prod_change_pred = np.diff(production_pred, prepend=production_pred[0])
    inj_change = np.diff(injection_rates, prepend=injection_rates[0])
    
    # Fit linear model: dQ_prod = efficiency * dQ_inj
    if inj_change.std() > EPSILON:
        eff_true = np.cov(prod_change_true, inj_change)[0, 1] / (inj_change.var() + EPSILON)
        eff_pred = np.cov(prod_change_pred, inj_change)[0, 1] / (inj_change.var() + EPSILON)
        
        metrics["injection_efficiency_true"] = eff_true
        metrics["injection_efficiency_pred"] = eff_pred
        metrics["injection_efficiency_error"] = abs(eff_pred - eff_true)
    
    # 3. Response time metric (lag between injection and production)
    # Use cross-correlation
    if len(production_true) > 10:
        corr = np.correlate(
            production_true - production_true.mean(),
            injection_rates - injection_rates.mean(),
            mode="full",
        )
        lags = np.arange(-len(production_true) + 1, len(production_true))
        positive_lags = lags >= 0
        
        best_lag_true = lags[positive_lags][np.argmax(corr[positive_lags])]
        
        corr_pred = np.correlate(
            production_pred - production_pred.mean(),
            injection_rates - injection_rates.mean(),
            mode="full",
        )
        best_lag_pred = lags[positive_lags][np.argmax(corr_pred[positive_lags])]
        
        metrics["response_lag_true"] = best_lag_true
        metrics["response_lag_pred"] = best_lag_pred
        metrics["response_lag_error"] = abs(best_lag_pred - best_lag_true)
    
    return metrics


def waterflood_performance_metrics(
    oil_rate_true: np.ndarray,
    oil_rate_pred: np.ndarray,
    water_cut_true: np.ndarray,
    water_cut_pred: np.ndarray,
) -> Dict[str, float]:
    """Metrics for waterflood performance.
    
    Args:
        oil_rate_true: Actual oil production rates
        oil_rate_pred: Predicted oil production rates
        water_cut_true: Actual water cut (fraction)
        water_cut_pred: Predicted water cut (fraction)
    
    Returns:
        Waterflood-specific metrics
    """
    metrics = {}
    
    # 1. Water breakthrough timing
    breakthrough_threshold = 0.5  # 50% water cut
    
    breakthrough_idx_true = np.where(water_cut_true >= breakthrough_threshold)[0]
    breakthrough_idx_pred = np.where(water_cut_pred >= breakthrough_threshold)[0]
    
    if len(breakthrough_idx_true) > 0 and len(breakthrough_idx_pred) > 0:
        metrics["breakthrough_time_true"] = breakthrough_idx_true[0]
        metrics["breakthrough_time_pred"] = breakthrough_idx_pred[0]
        metrics["breakthrough_time_error"] = abs(breakthrough_idx_pred[0] - breakthrough_idx_true[0])
    
    # 2. Water cut prediction error
    metrics["water_cut_mae"] = np.mean(np.abs(water_cut_pred - water_cut_true))
    metrics["water_cut_rmse"] = np.sqrt(np.mean((water_cut_pred - water_cut_true) ** 2))
    
    # 3. Oil recovery factor error
    # RF = cumulative oil / OOIP (original oil in place)
    # Approximate OOIP from early production
    if len(oil_rate_true) > 5:
        ooip_estimate = oil_rate_true[:5].mean() * 100  # Rough estimate
        
        cum_oil_true = np.cumsum(oil_rate_true)
        cum_oil_pred = np.cumsum(oil_rate_pred)
        
        rf_true = cum_oil_true[-1] / (ooip_estimate + EPSILON)
        rf_pred = cum_oil_pred[-1] / (ooip_estimate + EPSILON)
        
        metrics["recovery_factor_true"] = rf_true
        metrics["recovery_factor_pred"] = rf_pred
        metrics["recovery_factor_error"] = abs(rf_pred - rf_true)
    
    # 4. Sweep efficiency proxy (from water cut progression)
    # Faster water cut increase = lower sweep efficiency
    wc_gradient_true = np.gradient(water_cut_true)
    wc_gradient_pred = np.gradient(water_cut_pred)
    
    metrics["water_cut_gradient_true"] = wc_gradient_true.mean()
    metrics["water_cut_gradient_pred"] = wc_gradient_pred.mean()
    
    return metrics


def forecast_reliability_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_lower: Optional[np.ndarray] = None,
    y_pred_upper: Optional[np.ndarray] = None,
    confidence_level: float = 0.9,
) -> Dict[str, float]:
    """Metrics for forecast reliability and uncertainty quantification.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_pred_lower: Lower prediction interval (optional)
        y_pred_upper: Upper prediction interval (optional)
        confidence_level: Target confidence level (0-1)
    
    Returns:
        Reliability metrics
    """
    metrics = {}
    
    # 1. Direction accuracy (did we predict increase/decrease correctly?)
    if len(y_true) > 1:
        true_direction = np.sign(np.diff(y_true, prepend=y_true[0]))
        pred_direction = np.sign(np.diff(y_pred, prepend=y_pred[0]))
        
        direction_accuracy = (true_direction == pred_direction).mean()
        metrics["direction_accuracy"] = direction_accuracy
    
    # 2. Prediction interval coverage probability (PICP)
    if y_pred_lower is not None and y_pred_upper is not None:
        coverage = ((y_true >= y_pred_lower) & (y_true <= y_pred_upper)).mean()
        metrics["picp"] = coverage
        metrics["picp_deviation_from_target"] = abs(coverage - confidence_level)
        
        # 3. Mean Prediction Interval Width (MPIW)
        interval_width = y_pred_upper - y_pred_lower
        metrics["mpiw"] = interval_width.mean()
        
        # 4. Interval sharpness (normalized)
        y_range = y_true.max() - y_true.min()
        metrics["interval_sharpness"] = interval_width.mean() / (y_range + EPSILON)
    
    # 5. Forecast skill over persistence baseline
    if len(y_true) > 1:
        # Persistence forecast: next value = current value
        persistence_forecast = np.roll(y_true, 1)
        persistence_forecast[0] = y_true[0]
        
        mse_model = np.mean((y_true - y_pred) ** 2)
        mse_persistence = np.mean((y_true - persistence_forecast) ** 2)
        
        skill = 1 - mse_model / (mse_persistence + EPSILON)
        metrics["forecast_skill_vs_persistence"] = skill
    
    return metrics


def compute_all_reservoir_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_idx: Optional[np.ndarray] = None,
    pressure_true: Optional[np.ndarray] = None,
    pressure_pred: Optional[np.ndarray] = None,
    injection_rates: Optional[np.ndarray] = None,
    water_cut_true: Optional[np.ndarray] = None,
    water_cut_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive set of reservoir engineering metrics.
    
    Args:
        y_true: Actual production rates
        y_pred: Predicted production rates
        time_idx: Time indices
        pressure_true: Actual pressures (optional)
        pressure_pred: Predicted pressures (optional)
        injection_rates: Injection rates (optional)
        water_cut_true: Actual water cut (optional)
        water_cut_pred: Predicted water cut (optional)
    
    Returns:
        Dictionary with all computed metrics
    """
    all_metrics = {}
    
    # 1. Decline curve metrics
    decline_metrics = decline_curve_metrics(y_true, y_pred, time_idx)
    all_metrics.update({f"decline_{k}": v for k, v in decline_metrics.items()})
    
    # 2. Pressure metrics (if available)
    if pressure_true is not None and pressure_pred is not None:
        pressure_metrics = reservoir_pressure_metrics(pressure_true, pressure_pred, y_true)
        all_metrics.update({f"pressure_{k}": v for k, v in pressure_metrics.items()})
    
    # 3. Injection metrics (if available)
    if injection_rates is not None:
        injection_metrics = injection_efficiency_metrics(y_true, y_pred, injection_rates)
        all_metrics.update({f"injection_{k}": v for k, v in injection_metrics.items()})
    
    # 4. Waterflood metrics (if available)
    if water_cut_true is not None and water_cut_pred is not None:
        oil_rate_true = y_true * (1 - water_cut_true)
        oil_rate_pred = y_pred * (1 - water_cut_pred)
        
        waterflood_metrics = waterflood_performance_metrics(
            oil_rate_true, oil_rate_pred, water_cut_true, water_cut_pred
        )
        all_metrics.update({f"waterflood_{k}": v for k, v in waterflood_metrics.items()})
    
    # 5. Reliability metrics
    reliability_metrics = forecast_reliability_metrics(y_true, y_pred)
    all_metrics.update({f"reliability_{k}": v for k, v in reliability_metrics.items()})
    
    # Filter out None and non-finite values
    return {
        k: float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None
        for k, v in all_metrics.items()
    }
