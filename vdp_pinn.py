import argparse
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

def approx_period(mu):
    # for large mu, period is ~ 1.614 * mu
    return max(2.0 * math.pi, (3.0 - 2.0 * math.log(2.0)) * mu)

def reference_solution(mu, T, x0=2.0, v0=0.0, n_eval=4000):
    def rhs(t, y):
        x1, x2 = y
        return [x2, mu * (1.0 - x1 * x1) * x2 - x1]

    def jac(t, y):
        x1, x2 = y
        return [[0.0, 1.0],
                [-2.0 * mu * x1 * x2 - 1.0, mu * (1.0 - x1 * x1)]]

    t_eval = np.linspace(0.0, T, n_eval)
    sol = solve_ivp(
        rhs, (0.0, T), [x0, v0], method="Radau", jac=jac,
        t_eval=t_eval, rtol=1e-9, atol=1e-11, dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"Radau solver failed: {sol.message}")
    return sol.t, sol.y[0], sol.y[1], sol

class MLP(torch.nn.Module):
    def __init__(self, width=64, depth=4, n_out=2):
        super().__init__()
        layers = [torch.nn.Linear(1, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, n_out)]
        self.net = torch.nn.Sequential(*layers)
        
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1.0)
                torch.nn.init.zeros_(m.bias)

    def forward(self, s):
        return self.net(s)

def F_lienard(x):
    return x ** 3 / 3.0 - x

def lienard_ic(x0, v0, mu):
    y0 = v0 / mu + F_lienard(x0)
    return x0, y0

def lienard_to_velocity(x, y, mu):
    return mu * (y - F_lienard(x))

def _grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True,
    )[0]

# configurazione dei parametri
class TrainConfig:
    def __init__(self):
        self.width = 64
        self.depth = 4
        self.n_colloc = 256
        self.adam_iters = 2500
        self.adam_lr = 2e-3
        self.lbfgs_iters = 400
        self.verbose = False

def _make_ansatz(net, t, t_a, h, u0):
    s = 2.0 * (t - t_a) / h - 1.0
    n_out = net(s)
    u = u0 + (t - t_a) * n_out
    return u[:, 0:1], u[:, 1:2]

# funzione principale per allenare una singola finestra
def train_window(net, t_a, t_b, u0, mu, cfg):
    h = t_b - t_a
    x0_l = torch.tensor([[u0[0]]], device=DEVICE)
    y0_l = torch.tensor([[u0[1]]], device=DEVICE)
    u0_vec = torch.cat([x0_l, y0_l], dim=1)

    def sample_colloc():
        return torch.linspace(t_a, t_b, cfg.n_colloc, device=DEVICE).reshape(-1, 1)

    t_col = sample_colloc().requires_grad_(True)

    def residual_loss(t):
        x, y = _make_ansatz(net, t, t_a, h, u0_vec)
        x_t = _grad(x, t)
        y_t = _grad(y, t)
        
        R1 = x_t / mu - (y - F_lienard(x))
        R2 = mu * y_t + x
        return torch.mean(R1 ** 2) + torch.mean(R2 ** 2)

    opt = torch.optim.Adam(net.parameters(), lr=cfg.adam_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.adam_iters)
    
    # fase 1: addestramento con Adam
    for it in range(cfg.adam_iters):
        opt.zero_grad()
        loss = residual_loss(t_col)
        loss.backward()
        opt.step()
        sched.step()
        if cfg.verbose and (it % 500 == 0 or it == cfg.adam_iters - 1):
            print(f"      [Adam iter {it}] loss={loss.item():.3e}")

    if cfg.lbfgs_iters > 0:
        opt_lbfgs = torch.optim.LBFGS(
            net.parameters(), max_iter=cfg.lbfgs_iters, history_size=50,
            tolerance_grad=1e-12, tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_lbfgs.zero_grad()
            loss = residual_loss(t_col)
            loss.backward()
            return loss

        opt_lbfgs.step(closure)
        if cfg.verbose:
            final = residual_loss(t_col)
            print(f"      [LBFGS] loss={final.item():.3e}")

    with torch.no_grad():
        t_end = torch.tensor([[t_b]], device=DEVICE)
        s = 2.0 * (t_end - t_a) / h - 1.0
        u_end = u0_vec + (t_end - t_a) * net(s)
        return u_end[0, 0].item(), u_end[0, 1].item()

# metto i risultati in questa classe
class MarchResult:
    def __init__(self, t, x, v):
        self.t = t
        self.x = x
        self.v = v
        self.windows = []

def solve_time_marching(mu, T, x0=2.0, v0=0.0, h=None, cfg=None, eval_per_window=60):
    cfg = cfg or TrainConfig()
    if h is None:
        h = approx_period(mu) / 40.0
    
    n_windows = int(math.ceil(T / h))
    h = T / n_windows

    net = MLP(width=cfg.width, depth=cfg.depth).to(DEVICE)
    u0 = lienard_ic(x0, v0, mu)

    all_t, all_x, all_v = [], [], []
    t0 = time.time()
    
    for k in range(n_windows):
        t_a = k * h
        t_b = (k + 1) * h
        x_end, y_end = train_window(net, t_a, t_b, u0, mu, cfg)

        with torch.no_grad():
            ts = torch.linspace(t_a, t_b, eval_per_window, device=DEVICE).reshape(-1, 1)
            s = 2.0 * (ts - t_a) / h - 1.0
            u = torch.tensor([[u0[0], u0[1]]], device=DEVICE) + (ts - t_a) * net(s)
            xk = u[:, 0].cpu().numpy()
            yk = u[:, 1].cpu().numpy()
            vk = lienard_to_velocity(xk, yk, mu)
            
        all_t.append(ts.cpu().numpy().ravel())
        all_x.append(xk)
        all_v.append(vk)

        u0 = (x_end, y_end)
        if cfg.verbose or (k % max(1, n_windows // 10) == 0):
            print(f"  finestra {k+1}/{n_windows} completata. t in [{t_a:.3f},{t_b:.3f}]")

    print(f"  finito in {time.time() - t0:.1f} sec")
    
    t = np.concatenate(all_t)
    x = np.concatenate(all_x)
    v = np.concatenate(all_v)
    order = np.argsort(t)
    return MarchResult(t=t[order], x=x[order], v=v[order])

def _sample_window(net, t_a, h, u0, mu, n=60):
    with torch.no_grad():
        ts = torch.linspace(t_a, t_a + h, n, device=DEVICE).reshape(-1, 1)
        s = 2.0 * (ts - t_a) / h - 1.0
        u = torch.tensor([[u0[0], u0[1]]], device=DEVICE) + (ts - t_a) * net(s)
        xk = u[:, 0].cpu().numpy()
        yk = u[:, 1].cpu().numpy()
        vk = lienard_to_velocity(xk, yk, mu)
    return ts.cpu().numpy().ravel(), xk, yk, vk, float(np.max(np.abs(vk)))

def solve_time_marching_adaptive(mu, T, x0=2.0, v0=0.0, cfg=None, dx_cap=0.4, h_min=None, h_max=None, eval_per_window=60):
    cfg = cfg or TrainConfig()
    
    h_max = h_max or (approx_period(mu) / 15.0)
    h_min = h_min or (h_max / 64.0)

    net = MLP(width=cfg.width, depth=cfg.depth).to(DEVICE)
    u0 = lienard_ic(x0, v0, mu)

    all_t, all_x, all_v = [], [], []
    t_a = 0.0
    h = h_max
    n_win = 0
    n_retry = 0
    t0 = time.time()
    
    while t_a < T - 1e-9:
        h = min(h, T - t_a)
        snap = {k: v.detach().clone() for k, v in net.state_dict().items()}
        x_end, y_end = train_window(net, t_a, t_a + h, u0, mu, cfg)
        _, _, _, _, maxv = _sample_window(net, t_a, h, u0, mu, n=eval_per_window)

        if maxv * h > 2.0 * dx_cap and h > h_min * 1.5:
            net.load_state_dict(snap)
            h = max(h / 2.0, h_min)
            n_retry += 1
            continue

        ts, xk, yk, vk, _ = _sample_window(net, t_a, h, u0, mu, n=eval_per_window)
        all_t.append(ts); all_x.append(xk); all_v.append(vk)
        t_a += h
        u0 = (x_end, y_end)
        n_win += 1
        
        if cfg.verbose or n_win % 10 == 0:
            print(f"  finestra adattiva {n_win} a t={t_a:.3f}/{T:.2f} (passo h={h:.4f})")

        h = float(np.clip(dx_cap / (maxv + 1e-9), h_min, h_max))

    print(f"  marching adattivo completato in {time.time() - t0:.1f}s (totale: {n_win} finestre, {n_retry} retry)")

    t, x, v = np.concatenate(all_t), np.concatenate(all_x), np.concatenate(all_v)
    order = np.argsort(t)
    return MarchResult(t=t[order], x=x[order], v=v[order])

def solve_single_domain(mu, T, x0=2.0, v0=0.0, cfg=None, n_eval=2000):
    cfg = cfg or TrainConfig()
    net = MLP(width=cfg.width, depth=cfg.depth).to(DEVICE)
    u0 = lienard_ic(x0, v0, mu)
    
    train_window(net, 0.0, T, u0, mu, cfg)

    with torch.no_grad():
        ts = torch.linspace(0.0, T, n_eval, device=DEVICE).reshape(-1, 1)
        s = 2.0 * ts / T - 1.0
        u = torch.tensor([u0], device=DEVICE) + ts * net(s)
        x = u[:, 0].cpu().numpy()
        y = u[:, 1].cpu().numpy()
        v = lienard_to_velocity(x, y, mu)
        
    return ts.cpu().numpy().ravel(), x, v

def relative_l2_error(t_pinn, x_pinn, t_ref, x_ref):
    x_ref_i = np.interp(t_pinn, t_ref, x_ref)
    return np.sqrt(np.mean((x_pinn - x_ref_i) ** 2)) / np.sqrt(np.mean(x_ref_i ** 2))

def make_plots(mu, t_ref, x_ref, v_ref, t_pinn, x_pinn, v_pinn, mode, out_prefix):
    err = np.abs(x_pinn - np.interp(t_pinn, t_ref, x_ref))

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Van der Pol PINN ({mode}), mu = {mu}")

    axs[0, 0].plot(t_ref, x_ref, "k-", lw=1.2, label="Radau ref")
    axs[0, 0].plot(t_pinn, x_pinn, "r--", lw=1.0, label="PINN")
    axs[0, 0].set_xlabel("t"); axs[0, 0].set_ylabel("x(t)")
    axs[0, 0].legend(); axs[0, 0].set_title("x(t)")

    axs[0, 1].plot(t_ref, v_ref, "k-", lw=1.2, label="Radau ref")
    axs[0, 1].plot(t_pinn, v_pinn, "r--", lw=1.0, label="PINN")
    axs[0, 1].set_xlabel("t"); axs[0, 1].set_ylabel("v(t)")
    axs[0, 1].legend(); axs[0, 1].set_title("v(t)")

    axs[1, 0].plot(x_ref, v_ref, "k-", lw=1.0, label="Radau ref")
    axs[1, 0].plot(x_pinn, v_pinn, "r--", lw=1.0, label="PINN")
    axs[1, 0].set_xlabel("x"); axs[1, 0].set_ylabel("v")
    axs[1, 0].legend(); axs[1, 0].set_title("Phase")

    axs[1, 1].semilogy(t_pinn, err + 1e-16, "b-", lw=1.0)
    axs[1, 1].set_xlabel("t"); axs[1, 1].set_ylabel("err")
    axs[1, 1].set_title("Abs error (log)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"{out_prefix}_mu{mu:g}_{mode}.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    return fname

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--periods", type=float, default=2.0)
    p.add_argument("--T", type=float, default=None)
    p.add_argument("--mode", choices=["single", "march"], default="march")
    p.add_argument("--uniform", action="store_true")
    p.add_argument("--dx-cap", type=float, default=0.4)
    p.add_argument("--x0", type=float, default=2.0)
    p.add_argument("--v0", type=float, default=0.0)
    p.add_argument("--h", type=float, default=None)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--colloc", type=int, default=256)
    p.add_argument("--adam", type=int, default=2500)
    p.add_argument("--lbfgs", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="vdp")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)

    T = args.T if args.T is not None else args.periods * approx_period(args.mu)
    print(f"mu={args.mu} T={T:.2f} mode={args.mode}")

    t_ref, x_ref, v_ref, _ = reference_solution(args.mu, T, args.x0, args.v0)

    cfg = TrainConfig()
    cfg.width = args.width
    cfg.depth = args.depth
    cfg.n_colloc = args.colloc
    cfg.adam_iters = args.adam
    cfg.lbfgs_iters = args.lbfgs
    cfg.verbose = args.verbose

    if args.mode == "march":
        if args.uniform:
            res = solve_time_marching(args.mu, T, args.x0, args.v0, h=args.h, cfg=cfg)
        else:
            res = solve_time_marching_adaptive(args.mu, T, args.x0, args.v0, cfg=cfg, dx_cap=args.dx_cap)
        t_pinn, x_pinn, v_pinn = res.t, res.x, res.v
    else:
        t_pinn, x_pinn, v_pinn = solve_single_domain(args.mu, T, args.x0, args.v0, cfg=cfg)

    rel = relative_l2_error(t_pinn, x_pinn, t_ref, x_ref)
    print(f"L2 err: {rel:.3e}")
    
    fname = make_plots(args.mu, t_ref, x_ref, v_ref, t_pinn, x_pinn, v_pinn, args.mode, args.out)
    print(f"Saved: {fname}")

if __name__ == "__main__":
    main()
