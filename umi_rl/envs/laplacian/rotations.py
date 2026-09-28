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


def mat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] -> unit quaternion [..., 4] (w, x, y, z), Shepperd's method (batched, stable)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    cands = torch.stack([tr, m00, m11, m22], -1)
    case = cands.argmax(-1)
    def build(k):
        if k == 0:
            s = torch.sqrt((1 + tr).clamp_min(1e-12)) * 2
            return torch.stack([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s], -1)
        if k == 1:
            s = torch.sqrt((1 + m00 - m11 - m22).clamp_min(1e-12)) * 2
            return torch.stack([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s], -1)
        if k == 2:
            s = torch.sqrt((1 + m11 - m00 - m22).clamp_min(1e-12)) * 2
            return torch.stack([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s], -1)
        s = torch.sqrt((1 + m22 - m00 - m11).clamp_min(1e-12)) * 2
        return torch.stack([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s], -1)
    q = build(0)
    for k in (1, 2, 3):
        q = torch.where((case == k)[..., None], build(k), q)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    return torch.where((q[..., :1] < 0), -q, q)  # w >= 0 -> angle in [0, pi]


def mat_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] -> [..., 3] axis-angle (angle in [0, pi]), via the quaternion."""
    q = mat_to_quat_wxyz(R)
    w, xyz = q[..., 0].clamp(-1, 1), q[..., 1:]
    n = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(n.squeeze(-1), w)[..., None]
    scale = torch.where(n < 1e-8, 2.0 * torch.ones_like(n), angle / n.clamp_min(1e-12))
    return xyz * scale


def quat_wxyz_from_mat(R: torch.Tensor) -> torch.Tensor:
    return mat_to_quat_wxyz(R)
