from __future__ import annotations

from typing import Dict, Optional

import torch

EPSILON = 1e-6


def _ensure_3d(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None or not isinstance(value, torch.Tensor):
        return value
    if value.dim() == 2:
        return value.unsqueeze(0)
    return value


def crm_residual_loss(
    pred_prod: torch.Tensor,
    producer_forcing: Optional[torch.Tensor],
    bhp: Optional[torch.Tensor] = None,
    tau: Optional[torch.Tensor] = None,
    J: Optional[torch.Tensor] = None,
    Vp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if producer_forcing is None or pred_prod.numel() == 0:
        return pred_prod.new_zeros(())
    pred_prod = _ensure_3d(pred_prod)
    producer_forcing = _ensure_3d(producer_forcing)
    bhp = _ensure_3d(bhp)
    tau = _ensure_3d(tau)
    J = _ensure_3d(J)
    Vp = _ensure_3d(Vp)

    horizon = min(pred_prod.size(1), producer_forcing.size(1))
    if horizon <= 0:
        return pred_prod.new_zeros(())
    pred_prod = pred_prod[:, :horizon, :]
    drive = producer_forcing[:, :horizon, :]

    # Normalize forcing to the scale of predicted production before comparing dynamics.
    drive = drive / torch.clamp(drive.abs().mean(dim=1, keepdim=True), min=EPSILON)
    drive = drive * torch.clamp(pred_prod.abs().mean(dim=1, keepdim=True), min=EPSILON)
    if tau is not None:
        drive = drive / torch.clamp(tau[:, :horizon, :].abs(), min=1.0)
    if J is not None:
        drive = drive * torch.clamp(J[:, :horizon, :].abs(), min=0.1)
    if bhp is not None:
        drive = drive - 0.01 * bhp[:, :horizon, :]
    if Vp is not None:
        drive = drive / torch.clamp(Vp[:, :horizon, :].abs(), min=1.0)

    if horizon < 2:
        residual = pred_prod - drive
        return torch.mean(residual.pow(2))

    pred_delta = pred_prod[:, 1:, :] - pred_prod[:, :-1, :]
    drive_delta = drive[:, 1:, :] - drive[:, :-1, :]
    residual = pred_delta - drive_delta
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


def allocation_simplex_penalty(
    F_ij: Optional[torch.Tensor],
    *,
    group_index: Optional[torch.Tensor] = None,
    num_groups: Optional[int] = None,
) -> torch.Tensor:
    if F_ij is None or F_ij.numel() == 0:
        device = F_ij.device if isinstance(F_ij, torch.Tensor) else "cpu"
        return torch.zeros((), device=device)
    weights = _ensure_3d(F_ij)
    clipped = torch.clamp(weights, min=0.0)
    if group_index is not None:
        group_index = group_index.to(weights.device).long().view(-1)
        group_count = int(num_groups) if num_groups is not None else int(group_index.max().item()) + 1
        grouped = []
        for group_id in range(max(group_count, 0)):
            mask = group_index == group_id
            if torch.any(mask):
                grouped.append(clipped[..., mask].sum(dim=-1))
        if grouped:
            simplex_error = (torch.stack(grouped, dim=-1) - 1.0).pow(2)
        else:
            simplex_error = clipped.new_zeros(())
    else:
        simplex_error = (clipped.sum(dim=-1) - 1.0).pow(2)
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
    producer_forcing: Optional[torch.Tensor],
    *,
    alloc_weights: Optional[torch.Tensor] = None,
    alloc_history: Optional[torch.Tensor] = None,
    allocation_group_index: Optional[torch.Tensor] = None,
    allocation_num_groups: Optional[int] = None,
    shutin_mask: Optional[torch.Tensor] = None,
    bhp: Optional[torch.Tensor] = None,
    tau: Optional[torch.Tensor] = None,
    J: Optional[torch.Tensor] = None,
    Vp: Optional[torch.Tensor] = None,
    lambdas: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    lam = dict(lambdas or {})
    pred_cum = torch.cumsum(torch.clamp(pred_prod, min=0.0), dim=1) if pred_prod.dim() >= 2 else torch.cumsum(torch.clamp(pred_prod, min=0.0), dim=0)
    parts = {
        "crm_residual": crm_residual_loss(pred_prod, producer_forcing, bhp=bhp, tau=tau, J=J, Vp=Vp),
        "nonnegative": nonnegative_rate_penalty(pred_prod),
        "cumulative_monotonic": monotonic_cumulative_penalty(pred_cum),
        "allocation_simplex": allocation_simplex_penalty(
            alloc_weights,
            group_index=allocation_group_index,
            num_groups=allocation_num_groups,
        ),
        "shutin_consistency": shutin_consistency_penalty(pred_prod, shutin_mask),
        "temporal_smoothness": temporal_smoothness_penalty(alloc_history if alloc_history is not None else alloc_weights, J_t=J, Vp_t=Vp),
    }
    total = pred_prod.new_zeros(())
    for name, value in parts.items():
        total = total + float(lam.get(name, 1.0)) * value
    parts["total"] = total
    return parts
