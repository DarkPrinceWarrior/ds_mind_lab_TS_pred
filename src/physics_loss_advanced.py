"""Advanced Physics-Informed Loss Functions for reservoir modeling."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from neuralforecast.losses.pytorch import BasePointLoss, HuberLoss

logger = logging.getLogger(__name__)

EPSILON = 1e-6


class AdaptivePhysicsLoss(BasePointLoss):
    """Physics-informed loss with adaptive weighting and advanced reservoir physics.
    
    Research basis:
    - "WellPINN" (2025): Accurate well representation in PINNs
    - "Comprehensive review of physics-informed deep learning" (2025)
    
    Improvements over basic physics loss:
    1. Adaptive weight scheduling (increases during training)
    2. Multi-term physics: mass balance + diffusion + boundary conditions
    3. Heterogeneity awareness (per-well calibration)
    4. Soft constraints via penalty weighting
    """
    
    def __init__(
        self,
        base_loss: Optional[BasePointLoss] = None,
        physics_weight_init: float = 0.01,
        physics_weight_max: float = 0.3,
        injection_coeff: float = 0.05,
        damping: float = 0.01,
        diffusion_coeff: float = 0.001,
        smoothing_weight: float = 0.01,
        boundary_weight: float = 0.05,
        feature_names: Optional[List[str]] = None,
        adaptive_schedule: str = "linear",  # 'linear', 'exponential', 'cosine'
        warmup_steps: int = 50,
    ) -> None:
        if base_loss is None:
            base_loss = HuberLoss()
        
        super().__init__(
            horizon_weight=base_loss.horizon_weight,
            outputsize_multiplier=base_loss.outputsize_multiplier,
            output_names=base_loss.output_names,
        )
        
        self.base_loss = base_loss
        self.is_distribution_output = getattr(base_loss, "is_distribution_output", False)
        
        # Physics parameters
        self.physics_weight_init = float(physics_weight_init)
        self.physics_weight_max = float(physics_weight_max)
        self.injection_coeff = float(injection_coeff)
        self.damping = float(damping)
        self.diffusion_coeff = float(diffusion_coeff)
        self.smoothing_weight = float(smoothing_weight)
        self.boundary_weight = float(boundary_weight)
        
        # Adaptive scheduling
        self.adaptive_schedule = adaptive_schedule
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.current_physics_weight = physics_weight_init
        
        self.feature_names = list(feature_names) if feature_names else []
        self._context: Optional[Dict[str, torch.Tensor]] = None
        self.latest_terms: Dict[str, torch.Tensor] = {}
    
    def domain_map(self, y_hat: torch.Tensor) -> torch.Tensor:
        return self.base_loss.domain_map(y_hat)
    
    def set_context(
        self,
        injection: torch.Tensor,
        prev: torch.Tensor,
        mask: Optional[torch.Tensor],
        pressure: Optional[torch.Tensor] = None,
        spatial_features: Optional[torch.Tensor] = None,
    ) -> None:
        """Set context for physics computation."""
        self._context = {
            "injection": injection,
            "prev": prev,
            "mask": mask,
            "pressure": pressure,
            "spatial_features": spatial_features,
        }
    
    def clear_context(self) -> None:
        self._context = None
    
    def _update_physics_weight(self) -> float:
        """Adaptively update physics weight during training.
        
        Starts low to allow data fitting, increases to enforce physics.
        """
        if self.current_step < self.warmup_steps:
            progress = 0.0
        else:
            progress = min(1.0, (self.current_step - self.warmup_steps) / 200.0)
        
        if self.adaptive_schedule == "linear":
            weight = self.physics_weight_init + progress * (
                self.physics_weight_max - self.physics_weight_init
            )
        elif self.adaptive_schedule == "exponential":
            weight = self.physics_weight_init * (
                (self.physics_weight_max / self.physics_weight_init) ** progress
            )
        elif self.adaptive_schedule == "cosine":
            weight = self.physics_weight_init + 0.5 * (
                self.physics_weight_max - self.physics_weight_init
            ) * (1 - np.cos(np.pi * progress))
        else:
            weight = self.physics_weight_max
        
        self.current_physics_weight = float(weight)
        self.current_step += 1
        return weight
    
    def _mass_balance_penalty(
        self,
        y_hat: torch.Tensor,
        injection: torch.Tensor,
        prev: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mass balance: dQ/dt = alpha * Q_inj - beta * Q_prod.
        
        Research basis: Capacitance-Resistance Models (CRM)
        """
        # Convert to same device/dtype
        injection = injection.to(y_hat)
        prev = prev.to(y_hat)
        if mask is not None:
            mask = mask.to(y_hat)
        
        # Construct full history
        history = torch.cat([prev.unsqueeze(1), y_hat], dim=1)
        prev_steps = history[:, :-1, :]
        
        # Rate of change
        deltas = history[:, 1:, :] - prev_steps
        
        # Expected change from physics
        source_term = self.injection_coeff * injection
        sink_term = self.damping * prev_steps
        expected_delta = source_term - sink_term
        
        # Residual
        residual = deltas - expected_delta
        
        # Weighted penalty
        weight = torch.ones_like(residual) if mask is None else mask
        denom = torch.clamp(weight.sum(), min=EPSILON)
        penalty = torch.sum((residual ** 2) * weight) / denom
        
        return penalty
    
    def _diffusion_penalty(
        self,
        y_hat: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Diffusion penalty: penalize rapid changes (simulate pressure diffusion).
        
        Research basis: Diffusion equations in reservoir simulation
        """
        if y_hat.shape[1] < 2:
            return y_hat.new_zeros(())
        
        # Second derivative approximation
        d2y_dt2 = y_hat[:, 2:, :] - 2 * y_hat[:, 1:-1, :] + y_hat[:, :-2, :]
        
        # Diffusion equation residual: d2Q/dt2 should be small and smooth
        penalty = torch.mean(d2y_dt2 ** 2)
        
        return self.diffusion_coeff * penalty
    
    def _smoothness_penalty(
        self,
        y_hat: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Smoothness: penalize high-frequency oscillations."""
        if y_hat.shape[1] < 2:
            return y_hat.new_zeros(())
        
        # First derivative
        dy_dt = y_hat[:, 1:, :] - y_hat[:, :-1, :]
        
        # Second derivative (change in rate of change)
        if dy_dt.shape[1] < 2:
            return y_hat.new_zeros(())
        
        d2y_dt2 = dy_dt[:, 1:, :] - dy_dt[:, :-1, :]
        
        weight = torch.ones_like(d2y_dt2) if mask is None else mask[:, 2:, :]
        denom = torch.clamp(weight.sum(), min=EPSILON)
        penalty = torch.sum((d2y_dt2 ** 2) * weight) / denom
        
        return penalty
    
    def _boundary_penalty(
        self,
        y_hat: torch.Tensor,
        prev: torch.Tensor,
    ) -> torch.Tensor:
        """Boundary condition: initial forecast should be close to last observation.
        
        Ensures continuity at forecast boundary.
        """
        prev = prev.to(y_hat)
        
        # First prediction vs last observation
        initial_forecast = y_hat[:, 0, :]
        boundary_residual = (initial_forecast - prev) ** 2
        
        return torch.mean(boundary_residual)
    
    def _physics_residual(
        self,
        y_hat: torch.Tensor,
        injection: torch.Tensor,
        prev: torch.Tensor,
        mask: Optional[torch.Tensor],
        pressure: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined physics residual."""
        # Mass balance (primary term)
        mass_penalty = self._mass_balance_penalty(y_hat, injection, prev, mask)
        
        # Diffusion (secondary)
        diffusion_penalty = self._diffusion_penalty(y_hat, mask)
        
        # Smoothness
        smoothness_penalty = self._smoothness_penalty(y_hat, mask)
        
        # Boundary continuity
        boundary_penalty = self._boundary_penalty(y_hat, prev)
        
        # Combine
        total_physics = (
            mass_penalty
            + diffusion_penalty
            + self.smoothing_weight * smoothness_penalty
            + self.boundary_weight * boundary_penalty
        )
        
        # Store components for logging
        self.latest_terms.update({
            "mass_balance": mass_penalty.detach(),
            "diffusion": diffusion_penalty.detach(),
            "smoothness": smoothness_penalty.detach(),
            "boundary": boundary_penalty.detach(),
        })
        
        return total_physics
    
    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute data loss
        data_loss = self.base_loss(
            y=y,
            y_hat=y_hat,
            y_insample=y_insample,
            mask=mask,
        )
        
        # Compute physics penalty
        physics_penalty = data_loss.new_zeros(())
        
        if self._context is not None:
            ctx = self._context
            
            # Update adaptive weight
            current_weight = self._update_physics_weight()
            
            # Compute physics residual
            physics_penalty = self._physics_residual(
                y_hat=y_hat,
                injection=ctx["injection"],
                prev=ctx["prev"],
                mask=ctx.get("mask"),
                pressure=ctx.get("pressure"),
            )
            
            # Apply adaptive weight
            physics_penalty = current_weight * physics_penalty
        
        # Total loss
        total = data_loss + physics_penalty
        
        # Store for logging
        self.latest_terms["data"] = data_loss.detach()
        self.latest_terms["physics_total"] = physics_penalty.detach()
        self.latest_terms["physics_weight"] = torch.tensor(self.current_physics_weight)
        
        self.clear_context()
        return total


class EnsemblePhysicsLoss(BasePointLoss):
    """Ensemble multiple physics loss formulations.
    
    Research basis: Ensemble methods improve robustness
    """
    
    def __init__(
        self,
        base_loss: Optional[BasePointLoss] = None,
        loss_components: Optional[List[Dict]] = None,
    ) -> None:
        if base_loss is None:
            base_loss = HuberLoss()
        
        super().__init__(
            horizon_weight=base_loss.horizon_weight,
            outputsize_multiplier=base_loss.outputsize_multiplier,
            output_names=base_loss.output_names,
        )
        
        self.base_loss = base_loss
        
        # Create multiple physics loss instances with different params
        if loss_components is None:
            loss_components = [
                {"physics_weight_max": 0.1, "injection_coeff": 0.03, "damping": 0.01},
                {"physics_weight_max": 0.2, "injection_coeff": 0.05, "damping": 0.02},
                {"physics_weight_max": 0.3, "injection_coeff": 0.07, "damping": 0.015},
            ]
        
        self.losses = []
        for params in loss_components:
            loss = AdaptivePhysicsLoss(base_loss=base_loss, **params)
            self.losses.append(loss)
        
        logger.info("Created ensemble of %d physics loss functions", len(self.losses))
    
    def domain_map(self, y_hat: torch.Tensor) -> torch.Tensor:
        return self.base_loss.domain_map(y_hat)
    
    def set_context(self, **kwargs) -> None:
        for loss in self.losses:
            loss.set_context(**kwargs)
    
    def clear_context(self) -> None:
        for loss in self.losses:
            loss.clear_context()
    
    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute average of ensemble losses
        total_loss = 0.0
        
        for loss_fn in self.losses:
            loss_val = loss_fn(y=y, y_hat=y_hat, y_insample=y_insample, mask=mask)
            total_loss = total_loss + loss_val
        
        return total_loss / len(self.losses)
