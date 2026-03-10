import torch
import torch.nn.functional as F


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert real-first quaternions (..., 4) to rotation matrices (..., 3, 3)."""
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with shape (..., 4), got {quaternions.shape}")

    q = F.normalize(quaternions, dim=-1)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        (
            ww + xx - yy - zz,
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            ww - xx + yy - zz,
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            ww - xx - yy + zz,
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))

