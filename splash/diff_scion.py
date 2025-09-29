from __future__ import annotations
import torch
import torch.nn.functional as F

# ---- differentiable kernel ----------------------------------------------------

def pairwise_d2(X: torch.Tensor) -> torch.Tensor:
    # X: (T, d)
    # returns (T,T) squared distances
    x2 = (X * X).sum(-1, keepdim=True)
    d2 = x2 + x2.T - 2 * (X @ X.T)
    return d2.clamp_min(0.0)

def soft_knn_weights(d2: torch.Tensor, reach: float, k: int, tau: float = 0.2) -> torch.Tensor:
    """
    Differentiable 'soft kNN': we compute Gaussian weights then soften row-wise selection
    with a temperature tau (lower = sharper). Self-mass removed.
    """
    T = d2.size(0)
    W = torch.exp(-0.5 * d2 / (reach * reach + 1e-8))
    W.fill_diagonal_(0.0)
    # normalize to avoid explosion before soft select
    W = W / (W.sum(-1, keepdim=True) + 1e-12)

    # Soft top-k gating per row via temperature on normalized weights
    # This is a soft “keep biggest k” heuristic: sharpen distribution so
    # ~k neighbors dominate without hard masking.
    logits = torch.log(W + 1e-20) / tau
    A = F.softmax(logits, dim=-1)  # (T,T)
    # Encourage ~k neighbors by renormalizing to target mass k/(T-1)
    target_mass = float(k) / float(max(T - 1, 1))
    A = A * (target_mass / (A.sum(-1, keepdim=True) + 1e-12))

    # final row-normalized kernel
    Wn = A / (A.sum(-1, keepdim=True) + 1e-12)
    Wn.fill_diagonal_(0.0)
    return Wn

# ---- one step of differentiable 'dynamics' -----------------------------------

def scion_step_diff(X: torch.Tensor, theta: torch.Tensor, W: torch.Tensor, dt: float = 0.05):
    """
    Simplified differentiable phase update: theta_{t+1} = theta_t + dt * Laplacian(theta)
    Using W as row-stochastic kernel (random-walk).
    """
    # neighbor mean phase (wrapped via sin/cos to avoid angle discontinuity)
    cos_t = torch.cos(theta); sin_t = torch.sin(theta)
    cos_nb = W @ cos_t; sin_nb = W @ sin_t
    theta_nb = torch.atan2(sin_nb, cos_nb)  # mean direction
    dtheta = theta_nb - theta
    # wrap to (-pi,pi)
    dtheta = (dtheta + torch.pi) % (2 * torch.pi) - torch.pi
    theta_new = theta + dt * dtheta
    return theta_new

# ---- differentiable invariants -----------------------------------------------

def inv_alignment(theta: torch.Tensor) -> torch.Tensor:
    # R = |mean e^{i theta}|
    c = torch.cos(theta).mean()
    s = torch.sin(theta).mean()
    return torch.sqrt(c * c + s * s + 1e-12)

def inv_curvature_proxy(theta: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # tension-like: mean 1 - cos(theta_j - theta_i) over edges
    T = theta.shape[0]
    # expected neighbor difference per node
    cos_nb = torch.cos(theta.unsqueeze(1) - theta.unsqueeze(0))  # (T,T)
    val = (W * (1.0 - cos_nb)).sum() / (W.sum() + 1e-12)
    return val

def inv_energy(theta: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # simple energy proxy: sum (1 - cos(theta_j - theta_i)) * W_ij
    cos_nb = torch.cos(theta.unsqueeze(1) - theta.unsqueeze(0))
    return (W * (1.0 - cos_nb)).sum()

def inv_dislocation_proxy(theta: torch.Tensor) -> torch.Tensor:
    # phase roughness via second difference (1D surrogate on token axis)
    d1 = theta[1:] - theta[:-1]
    d1 = (d1 + torch.pi) % (2*torch.pi) - torch.pi
    d2 = d1[1:] - d1[:-1]
    d2 = (d2 + torch.pi) % (2*torch.pi) - torch.pi
    return (d2.abs().mean())

def inv_flux(prev_E: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    return E - prev_E

# ---- unrolled differentiable measures ----------------------------------------

def unroll_measures_diff(
    X: torch.Tensor,               # (T,d) places (L2 or cosine-prep)
    k: int = 8,
    reach: float = 1.5,
    steps: int = 8,
    dt: float = 0.05,
    tau: float = 0.2,
) -> dict:
    """
    Build soft-kNN kernel, unroll a few steps, and return differentiable invariants.
    """
    T = X.size(0)
    d2 = pairwise_d2(X)
    W = soft_knn_weights(d2, reach=reach, k=k, tau=tau)  # (T,T)
    # init phase from a stable seed (PCA on X is possible; uniform random is fine with unroll)
    with torch.no_grad():
        theta0 = torch.zeros(T, device=X.device)  # or random small noise
    theta = theta0.clone().requires_grad_(True)

    E_prev = inv_energy(theta, W)
    for _ in range(steps):
        theta = scion_step_diff(X, theta, W, dt=dt)
    R = inv_alignment(theta)
    Q = inv_curvature_proxy(theta, W)
    E = inv_energy(theta, W)
    rho = inv_dislocation_proxy(theta)
    PhiE = inv_flux(E_prev, E)
    return {"R": R, "Q": Q, "E": E, "rho_phi": rho, "PhiE": PhiE}
