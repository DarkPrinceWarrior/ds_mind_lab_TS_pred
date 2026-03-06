from __future__ import annotations

from typing import Dict, Optional

import torch

EPSILON = 1e-6


def crm_residual_loss(
    pred_prod: torch.Tensor,
    inj_rates: Optional[torch.Tensor],
    bhp: Optional[torch.Tensor] = None,
    alloc_weights: Optional[torch.Tensor] = None,
    tau: Optional[torch.Tensor] = None,
    J: Optional[torch.Tensor] = None,
    Vp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if inj_rates is None or pred_prod.numel() == 0:
        return pred_prod.new_zeros(())
    if pred_prod.dim() == 2:
        pred_prod = pred_prod.unsqueeze(0)
    if inj_rates.dim() == 2:
        inj_rates = inj_rates.unsqueeze(0)

    inj_signal = inj_rates.mean(dim=-1, keepdim=True)
    pred_delta = pred_prod[:, 1:, :] - pred_prod[:, :-1, :] if pred_prod.size(1) > 1 else pred_prod.new_zeros(())
    inj_delta = inj_signal[:, 1:, :] - inj_signal[:, :-1, :] if inj_signal.size(1) > 1 else inj_signal.new_zeros(())
    if alloc_weights is not None:
        alloc_term = alloc_weights.mean(dim=-1, keepdim=True)
        inj_delta = inj_delta * alloc_term[:, : inj_delta.size(1), :]
    if tau is not None:
        inj_delta = inj_delta / torch.clamp(tau.abs().mean(), min=1.0)
    if J is not None:
        inj_delta = inj_delta * torch.clamp(J.mean().abs(), min=0.1)
    if bhp is not None:
        inj_delta = inj_delta - 0.01 * bhp[:, 1:, :].mean(dim=-1, keepdim=True)
    if Vp is not None:
        inj_delta = inj_delta / torch.clamp(Vp.mean().abs(), min=1.0)
    if pred_delta.numel() == 0 or inj_delta.numel() == 0:
        return pred_prod.new_zeros(())
    residual = pred_delta - inj_delta.expand_as(pred_delta)
    return torch.mean(residual.pow(2))


def nonnegative_rate_penalty(pred_prod: torch.Tensor) -> torch.Tensor:
    return torch.relu(-pred_prod).mean()


def monotonic_cumulative_penalty(pred_cum: torch.Tensor) -> torch.Tensor:
    if pred_cum.size(-1) < 2 and pred_cum.dim() >= 2 and pred_cum.size(1) >= 2:
        diffs = pred_cum[:, 1:, ...] - pred_cum[:, :-1, ...]
    elif pred_cum.dim() >= 2 and pred_cum.size(1) >= 2:
        diffs = pred_cum[:, 1:, ...] - pred_cum[:, :-1, ...]
    else:
        return pred_cum.new_zeros(())
    return torch.relu(-diffs).mean()


def allocation_simplex_penalty(F_ij: Optional[torch.Tensor]) -> torch.Tensor:
    if F_ij is None or F_ij.numel() == 0:
        device = F_ij.device if isinstance(F_ij, torch.Tensor) else "cpu"
        return torch.zeros((), device=device)
    weights = torch.clamp(F_ij, min=0.0)
    simplex_error = (weights.sum(dim=-1) - 1.0).pow(2)
    negative_error = torch.relu(-F_ij).pow(2)
    return simplex_error.mean() + negative_error.mean()


def shutin_consistency_penalty(
    pred_prod: torch.Tensor,
    scenario_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if scenario_mask is None or scenario_mask.numel() == 0:
        return pred_prod.new_zeros(())
    mask = scenario_mask.to(pred_prod).bool()
    if mask.shape != pred_prod.shape:
        mask = mask.expand_as(pred_prod)
    return pred_prod.masked_select(mask).abs().mean() if mask.any() else pred_prod.new_zeros(())


def temporal_smoothness_penalty(
    F_ij_t: Optional[torch.Tensor],
    J_t: Optional[torch.Tensor] = None,
    Vp_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if F_ij_t is None or F_ij_t.numel() == 0 or F_ij_t.size(1) < 2:
        device = F_ij_t.device if isinstance(F_ij_t, torch.Tensor) else "cpu"
        return torch.zeros((), device=device)
    diffs = F_ij_t[:, 1:, ...] - F_ij_t[:, :-1, ...]
    penalty = diffs.pow(2).mean()
    if J_t is not None and J_t.numel() > 0:
        penalty = penalty * torch.clamp(J_t.abs().mean(), min=0.1)
    if Vp_t is not None and Vp_t.numel() > 0:
        penalty = penalty / torch.clamp(Vp_t.abs().mean(), min=1.0)
    return penalty


def build_physics_loss_breakdown(
    pred_prod: torch.Tensor,
    inj_rates: Optional[torch.Tensor],
    *,
    alloc_weights: Optional[torch.Tensor] = None,
    shutin_mask: Optional[torch.Tensor] = None,
    tau: Optional[torch.Tensor] = None,
    J: Optional[torch.Tensor] = None,
    Vp: Optional[torch.Tensor] = None,
    lambdas: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    lam = dict(lambdas or {})
    pred_cum = torch.cumsum(torch.clamp(pred_prod, min=0.0), dim=1) if pred_prod.dim() >= 2 else torch.cumsum(torch.clamp(pred_prod, min=0.0), dim=0)
    parts = {
        "crm_residual": crm_residual_loss(pred_prod, inj_rates, alloc_weights=alloc_weights, tau=tau, J=J, Vp=Vp),
        "nonnegative": nonnegative_rate_penalty(pred_prod),
        "cumulative_monotonic": monotonic_cumulative_penalty(pred_cum),
        "allocation_simplex": allocation_simplex_penalty(alloc_weights),
        "shutin_consistency": shutin_consistency_penalty(pred_prod, shutin_mask),
        "temporal_smoothness": temporal_smoothness_penalty(alloc_weights),
    }
    total = pred_prod.new_zeros(())
    for name, value in parts.items():
        total = total + float(lam.get(name, 1.0)) * value
    parts["total"] = total
    return parts
