"""Small batched SO(3) helpers (torch), matching scipy's rotvec convention."""
from __future__ import annotations

import torch


def rotvec_to_mat(v: torch.Tensor) -> torch.Tensor:
    """[..., 3] axis-angle -> [..., 3, 3] via Rodrigues."""
    theta = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-12)
    k = v / theta
    K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(K)
    small = (theta < 1e-8)[..., None]
    R = eye + s * K + (1 - c) * (K @ K)
    return torch.where(small, eye + K * theta[..., None], R)


def mat_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] -> [..., 3] axis-angle (angle in [0, pi])."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1) / 2).clamp(-1, 1)
    theta = torch.acos(cos)
    axis = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    sin = torch.sin(theta)
    generic = axis / (2 * sin.clamp_min(1e-12))[..., None] * theta[..., None]
    small = theta < 1e-6
    # near pi: axis from the symmetric part
    near_pi = theta > 3.1
    S = (R + R.transpose(-1, -2)) / 2
    diag = torch.stack([S[..., 0, 0], S[..., 1, 1], S[..., 2, 2]], -1)
    ax = torch.sqrt(((diag + 1) / 2).clamp_min(0))
    # fix signs using the off-diagonal terms relative to the largest component
    idx = diag.argmax(-1)
    sgn = torch.ones_like(ax)
    for j in range(3):
        for i in range(3):
            if i != j:
                sgn[..., i] = torch.where(idx == j, torch.sign(S[..., i, j] + 1e-20), sgn[..., i])
    ax = ax * sgn
    ax = ax / torch.linalg.vector_norm(ax, dim=-1, keepdim=True).clamp_min(1e-12)
    out = torch.where(small[..., None], axis / 2, generic)
    return torch.where(near_pi[..., None], ax * theta[..., None], out)


def quat_wxyz_from_mat(R: torch.Tensor) -> torch.Tensor:
    v = mat_to_rotvec(R)
    theta = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    axis = v / theta.clamp_min(1e-12)
    return torch.cat([torch.cos(theta / 2), axis * torch.sin(theta / 2)], -1)
