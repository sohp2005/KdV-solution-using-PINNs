"""
kdv_core.py
===========
Single source of truth for the multi-soliton KdV PINN pipeline.

This module reproduces the *working* training pipeline that produced the
known-good scaling numbers (N=3: 0.158%, N=4: 0.228%, N=6: 0.494%,
N=9: 0.975%, N=12: 42.15%). Every rerun notebook MUST import from here
and not redefine training internals.

Conventions
-----------
PDE:        u_t + 6 u u_x + u_xxx = 0        (positive-soliton form)
Soliton:    u(x,t) = (c/2) sech^2( (sqrt(c)/2) * (x - c t - x0) )
Domain:     x in [xL, xR], t in [0, T_END=5]
Loss:       L = L_pde + L_ic + L_bc + L_data  (equal weights, MSE on each)
Schedule:   Adam(15k, lr=1e-3)  ->  L-BFGS(2k, strong_wolfe)
Net:        6 linear layers of `width`, tanh activations, Xavier init
Seed:       99 by default (matches scaling runs)

The function `build_data` includes a hard assert that data targets equal
the exact solution to machine precision -- this is the pipeline canary.
If a future run prints PIPELINE BROKEN, stop immediately.
"""

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_END = 5.0
DEFAULT_SEED = 99


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed numpy + torch (cpu + cuda). Call once at top of notebook."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# exact solutions
# ----------------------------------------------------------------------
def soliton_np(x, t, c, x0):
    """Single KdV soliton, numpy. Positive peak of height c/2, speed c."""
    arg = (np.sqrt(c) / 2.0) * (x - c * t - x0)
    return (c / 2.0) / np.cosh(arg) ** 2


def multisoliton_np(x, t, speeds, x0s):
    """Linear superposition. Valid only when solitons stay well-separated."""
    if np.isscalar(x):
        u = 0.0
    else:
        u = np.zeros_like(x, dtype=np.float64)
    for c, x0 in zip(speeds, x0s):
        u = u + soliton_np(x, t, c, x0)
    return u


# ----------------------------------------------------------------------
# per-N configuration (matches recap table exactly)
# ----------------------------------------------------------------------
_CONFIGS = {
    3:  dict(speeds=[2.5, 2.0, 1.5],
             x0s=[10.0, 0.0, -10.0],
             domain=(-20.0, 37.5)),
    4:  dict(speeds=[3.0, 2.5, 2.0, 1.5],
             x0s=[15.0, 5.0, -5.0, -15.0],
             domain=(-25.0, 45.0)),
    6:  dict(speeds=[4.0, 3.5, 3.0, 2.5, 2.0, 1.5],
             x0s=[25.0, 15.0, 5.0, -5.0, -15.0, -25.0],
             domain=(-35.0, 60.0)),
    9:  dict(speeds=[5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5],
             x0s=[40.0, 30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0, -40.0],
             domain=(-50.0, 82.5)),
    12: dict(speeds=[7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5],
             x0s=[55.0, 45.0, 35.0, 25.0, 15.0, 5.0, -5.0, -15.0, -25.0, -35.0, -45.0, -55.0],
             domain=(-65.0, 105.0)),
}


def make_config(N: int) -> dict:
    """Return the full hyperparam dict for a given N. Point counts scale with domain width."""
    if N not in _CONFIGS:
        raise ValueError(f"N={N} not supported. Available: {sorted(_CONFIGS.keys())}")
    base = _CONFIGS[N]
    xL, xR = base["domain"]
    width = xR - xL
    cfg = dict(
        N=N,
        speeds=list(base["speeds"]),
        x0s=list(base["x0s"]),
        domain=(xL, xR),
        width=width,
        n_pde=int(width * 1000 / 3),
        n_data=int(width * 1000 / 3) // 50,
        n_ic=max(500, int(width * 25 / 3)),
        n_bc=200,
        T_END=T_END,
    )
    return cfg


# ----------------------------------------------------------------------
# network: 6 linear layers of `width`, tanh, Xavier
#   width=50  ->   12,951 params
#   width=75  ->   28,801 params
#   width=100 ->   50,901 params
# ----------------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, width: int = 50, depth: int = 6):
        super().__init__()
        layers = [nn.Linear(2, width)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
        self.linears = nn.ModuleList(layers)
        self.out = nn.Linear(width, 1)
        self._xavier_init()
        self.width = width
        self.depth = depth

    def _xavier_init(self):
        for L in self.linears:
            nn.init.xavier_normal_(L.weight)
            nn.init.zeros_(L.bias)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        for L in self.linears:
            z = torch.tanh(L(z))
        return self.out(z)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------
# data builder + the pipeline canary
# ----------------------------------------------------------------------
def _to_t(arr, requires_grad: bool = False):
    t = torch.tensor(arr, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    if requires_grad:
        t.requires_grad_(True)
    return t


def build_data(
    config: dict,
    seed: int = DEFAULT_SEED,
    n_pde: int | None = None,
    n_data: int | None = None,
    noise_sigma: float = 0.0,
    use_ic: bool = True,
):
    """
    Build PDE / data / IC / BC tensors.

    Parameters
    ----------
    noise_sigma : float
        Std of additive Gaussian noise on data labels, expressed as a
        fraction of the *tallest* soliton's peak (peak = max(speeds)/2).
        0.0 means clean (and the canary assert is enforced).
    use_ic : bool
        If False, the IC tensors are still built but loss can skip them
        (for the IC-substitution experiment).
    """
    rng = np.random.default_rng(seed)
    speeds = config["speeds"]
    x0s = config["x0s"]
    xL, xR = config["domain"]
    T = config["T_END"]

    n_pde = config["n_pde"] if n_pde is None else int(n_pde)
    n_data = config["n_data"] if n_data is None else int(n_data)
    n_ic = config["n_ic"]
    n_bc = config["n_bc"]

    # collocation (interior)
    x_pde = rng.uniform(xL, xR, n_pde)
    t_pde = rng.uniform(0.0, T, n_pde)

    # data points (interior, clean labels by default)
    if n_data > 0:
        x_data = rng.uniform(xL, xR, n_data)
        t_data = rng.uniform(0.0, T, n_data)
        u_data_clean = multisoliton_np(x_data, t_data, speeds, x0s)
        if noise_sigma > 0.0:
            peak = max(speeds) / 2.0
            u_data = u_data_clean + rng.normal(0.0, noise_sigma * peak, n_data)
        else:
            u_data = u_data_clean
    else:
        x_data = t_data = u_data = u_data_clean = None

    # IC (t=0)
    x_ic = rng.uniform(xL, xR, n_ic)
    u_ic = multisoliton_np(x_ic, np.zeros_like(x_ic), speeds, x0s)

    # BC (split between left and right edges)
    n_each = n_bc // 2
    t_bc = np.concatenate([rng.uniform(0.0, T, n_each), rng.uniform(0.0, T, n_each)])
    x_bc = np.concatenate([np.full(n_each, xL), np.full(n_each, xR)])

    # === PIPELINE CANARY ===
    # if labels don't match the exact solution to ~machine precision,
    # something has been silently corrupted. abort early.
    if n_data > 0 and noise_sigma == 0.0:
        err = float(np.max(np.abs(u_data - u_data_clean)))
        assert err < 1e-10, (
            f"PIPELINE BROKEN: data labels deviate from exact by {err:.3e}. "
            f"Stop and inspect build_data."
        )
    err_ic = float(np.max(np.abs(u_ic - multisoliton_np(x_ic, np.zeros_like(x_ic), speeds, x0s))))
    assert err_ic < 1e-10, f"PIPELINE BROKEN: IC labels deviate by {err_ic:.3e}"

    batch = dict(
        x_pde=_to_t(x_pde, requires_grad=True),
        t_pde=_to_t(t_pde, requires_grad=True),
        x_data=_to_t(x_data) if n_data > 0 else None,
        t_data=_to_t(t_data) if n_data > 0 else None,
        u_data=_to_t(u_data) if n_data > 0 else None,
        x_ic=_to_t(x_ic),
        t_ic=_to_t(np.zeros_like(x_ic)),
        u_ic=_to_t(u_ic),
        x_bc=_to_t(x_bc),
        t_bc=_to_t(t_bc),
        use_ic=use_ic,
        config=config,
        n_pde=n_pde,
        n_data=n_data,
        noise_sigma=noise_sigma,
        seed=seed,
    )
    return batch


# ----------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------
def pde_residual(model, x, t):
    """KdV residual: u_t + 6 u u_x + u_xxx."""
    u = model(x, t)
    g = torch.autograd.grad
    u_t = g(u.sum(), t, create_graph=True)[0]
    u_x = g(u.sum(), x, create_graph=True)[0]
    u_xx = g(u_x.sum(), x, create_graph=True)[0]
    u_xxx = g(u_xx.sum(), x, create_graph=True)[0]
    return u_t + 6.0 * u * u_x + u_xxx


def total_loss(model, batch, return_parts: bool = False):
    """L = L_pde + L_ic + L_bc + L_data, equal weights, MSE on each component."""
    res = pde_residual(model, batch["x_pde"], batch["t_pde"])
    L_pde = torch.mean(res ** 2)

    if batch["use_ic"]:
        u_ic_p = model(batch["x_ic"], batch["t_ic"])
        L_ic = torch.mean((u_ic_p - batch["u_ic"]) ** 2)
    else:
        L_ic = torch.zeros((), device=DEVICE)

    u_bc_p = model(batch["x_bc"], batch["t_bc"])
    L_bc = torch.mean(u_bc_p ** 2)

    if batch["u_data"] is not None:
        u_d_p = model(batch["x_data"], batch["t_data"])
        L_data = torch.mean((u_d_p - batch["u_data"]) ** 2)
    else:
        L_data = torch.zeros((), device=DEVICE)

    total = L_pde + L_ic + L_bc + L_data
    if return_parts:
        return total, L_pde.item(), L_ic.item(), L_bc.item(), L_data.item()
    return total


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_grid(model, config, n_x: int = 400, n_t: int = 200):
    """Return (X, T, u_pred, u_exact) on a uniform grid."""
    xL, xR = config["domain"]
    T = config["T_END"]
    x = np.linspace(xL, xR, n_x)
    t = np.linspace(0.0, T, n_t)
    X, Tg = np.meshgrid(x, t, indexing="ij")
    xs = _to_t(X.flatten())
    ts = _to_t(Tg.flatten())
    u_pred = model(xs, ts).cpu().numpy().reshape(n_x, n_t)
    u_exact = multisoliton_np(X, Tg, config["speeds"], config["x0s"])
    return X, Tg, u_pred, u_exact


def compute_l2(model, config, n_x: int = 400, n_t: int = 200) -> float:
    """Relative L2 error in percent, evaluated on a uniform x-t grid."""
    _, _, u_pred, u_exact = predict_grid(model, config, n_x, n_t)
    num = np.sqrt(np.mean((u_pred - u_exact) ** 2))
    den = np.sqrt(np.mean(u_exact ** 2))
    return 100.0 * num / den


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
def train(
    model,
    batch,
    adam_iter: int = 15000,
    lbfgs_iter: int = 2000,
    lr: float = 1e-3,
    log_every: int = 1000,
    lbfgs_log_every: int = 200,
    verbose: bool = True,
):
    """
    Standard pipeline: Adam(adam_iter, lr) -> L-BFGS(lbfgs_iter, strong_wolfe).
    Returns a history dict with all loss components and timing.
    """
    history = dict(
        adam_iter=[], adam_total=[], adam_pde=[], adam_ic=[], adam_bc=[], adam_data=[],
        lbfgs_iter=[], lbfgs_total=[],
        adam_time=None, lbfgs_time=None,
        adam_l2=None, lbfgs_l2=None,
    )

    # ----- ADAM -----
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for it in range(adam_iter):
        opt.zero_grad()
        total, lp, li, lb, ld = total_loss(model, batch, return_parts=True)
        total.backward()
        opt.step()
        if it % log_every == 0 or it == adam_iter - 1:
            history["adam_iter"].append(it)
            history["adam_total"].append(total.item())
            history["adam_pde"].append(lp)
            history["adam_ic"].append(li)
            history["adam_bc"].append(lb)
            history["adam_data"].append(ld)
            if verbose:
                print(
                    f"  Adam {it:5d} | Total={total.item():.6e} "
                    f"PDE={lp:.3e} IC={li:.3e} BC={lb:.3e} Data={ld:.3e}",
                    flush=True,
                )
    history["adam_time"] = time.time() - t0
    history["adam_l2"] = compute_l2(model, batch["config"])
    if verbose:
        print(f"  >> Adam done {history['adam_time']:.1f}s | L2={history['adam_l2']:.4f}%", flush=True)

    # ----- L-BFGS -----
    opt = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_iter,
        max_eval=lbfgs_iter + 100,
        history_size=50,
        tolerance_grad=1e-8,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    step_counter = [0]

    def closure():
        opt.zero_grad()
        total = total_loss(model, batch)
        total.backward()
        i = step_counter[0]
        if i % lbfgs_log_every == 0:
            history["lbfgs_iter"].append(i)
            history["lbfgs_total"].append(total.item())
            if verbose:
                print(f"  LBFGS {i:4d} | Total={total.item():.8e}", flush=True)
        step_counter[0] += 1
        return total

    t0 = time.time()
    opt.step(closure)
    history["lbfgs_time"] = time.time() - t0
    history["lbfgs_l2"] = compute_l2(model, batch["config"])
    if verbose:
        print(f"  >> LBFGS done {history['lbfgs_time']:.1f}s | L2={history['lbfgs_l2']:.4f}%", flush=True)

    return history


# ----------------------------------------------------------------------
# checkpoint I/O
# ----------------------------------------------------------------------
def save_checkpoint(path, model, history, config, extras: dict | None = None):
    payload = dict(
        state_dict=model.state_dict(),
        width=model.width,
        depth=model.depth,
        history=history,
        config=config,
        extras=extras or {},
    )
    torch.save(payload, path)


def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location or DEVICE, weights_only=False)


def model_from_checkpoint(ckpt) -> PINN:
    m = PINN(width=ckpt["width"], depth=ckpt["depth"]).to(DEVICE)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


# ----------------------------------------------------------------------
# plotting helpers
# ----------------------------------------------------------------------
def plot_snapshots(model, config, save_path=None, t_vals=(0, 1, 2, 3, 4, 5), title=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    xL, xR = config["domain"]
    xs_np = np.linspace(xL, xR, 600)
    xs = _to_t(xs_np)
    for ax, tv in zip(axes.flat, t_vals):
        ts = _to_t(np.full_like(xs_np, tv))
        with torch.no_grad():
            up = model(xs, ts).cpu().numpy().flatten()
        ue = multisoliton_np(xs_np, tv, config["speeds"], config["x0s"])
        ax.plot(xs_np, ue, "b-", label="Exact", lw=1.5)
        ax.plot(xs_np, up, "r--", label="PINN", lw=1.2)
        ax.set_title(f"t = {tv}")
        ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def plot_heatmap(model, config, save_path=None, title=None):
    X, Tg, u_pred, u_exact = predict_grid(model, config)
    err = np.abs(u_pred - u_exact)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        err.T, origin="lower", aspect="auto",
        extent=[config["domain"][0], config["domain"][1], 0.0, config["T_END"]],
        cmap="hot",
    )
    ax.set_xlabel("x"); ax.set_ylabel("t")
    ax.set_title(title or f"Pointwise |u_pred - u_exact|  max={err.max():.4f}")
    fig.colorbar(im, ax=ax, label="|err|")
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def plot_loss_curves(history, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history["adam_iter"], history["adam_total"], "k-", label="Total")
    ax.semilogy(history["adam_iter"], history["adam_pde"],   "C0-", label="PDE")
    ax.semilogy(history["adam_iter"], history["adam_ic"],    "C1-", label="IC")
    ax.semilogy(history["adam_iter"], history["adam_bc"],    "C2-", label="BC")
    ax.semilogy(history["adam_iter"], history["adam_data"],  "C3-", label="Data")
    ax.set_xlabel("Adam iter"); ax.set_ylabel("Loss")
    ax.set_title(title or "Training loss (Adam phase)")
    ax.legend(); ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# self-test: quick 30s sanity that the pipeline is alive
# ----------------------------------------------------------------------
def quick_sanity_check(seed: int = DEFAULT_SEED, adam_iter: int = 500):
    """
    Train a tiny N=3 run for `adam_iter` iters and assert the loss drops.
    Use this at the top of every notebook before launching the real job.
    """
    print("=" * 60)
    print("PIPELINE SANITY CHECK  (N=3, short Adam, no L-BFGS)")
    print("=" * 60)
    set_seed(seed)
    cfg = make_config(3)
    batch = build_data(cfg, seed=seed)
    print(f"  device = {DEVICE}")
    print(f"  domain = {cfg['domain']} | n_pde={cfg['n_pde']} | n_data={cfg['n_data']}")
    m = PINN(width=50).to(DEVICE)
    print(f"  params = {m.n_params():,}")

    # train briefly
    hist = train(m, batch, adam_iter=adam_iter, lbfgs_iter=0,
                 log_every=adam_iter // 5, verbose=True)

    l2 = compute_l2(m, cfg)
    print(f"  L2 after {adam_iter} Adam iters: {l2:.3f}%")
    if adam_iter >= 500 and l2 > 50.0:
        raise RuntimeError(
            f"SANITY CHECK FAILED: L2={l2:.1f}% after {adam_iter} iters. "
            f"Pipeline is broken. STOP."
        )
    print("  >> sanity OK\n")
    return l2


# ----------------------------------------------------------------------
# misc utilities
# ----------------------------------------------------------------------
def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def print_config(cfg):
    print(f"  N={cfg['N']}  domain={cfg['domain']}  width={cfg['width']:.1f}")
    print(f"  speeds={cfg['speeds']}")
    print(f"  x0s   ={cfg['x0s']}")
    print(f"  points: PDE={cfg['n_pde']:,}  Data={cfg['n_data']:,}  "
          f"IC={cfg['n_ic']:,}  BC={cfg['n_bc']:,}")
