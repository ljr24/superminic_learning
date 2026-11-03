"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mdp import INV_DIM, NUM_J, XI_DIM, invariant_features
from .rotations import so3_exp, yaw_mat


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


# 统计量是buffer，不是优化器训练的参数；采集时更新统计，预测时仍可对输入求导。
class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


# 输出70维：局部位移3、旋转向量3、线/角速度增量3+3、关节角/速度增量29+29。
WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J

# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


# 【世界模型阅读入口：论文§3.2、表A.4；以下具体坐标/归一化设计以本仓库为准】
# S=(p,R,v,w,q,qd)：p/v/w形状(B,3)，R为(B,3,3)，q/qd为(B,29)。
# R将身体坐标转到世界坐标；v/w是世界坐标中的根部线/角速度。
# a形状(B,29)，xi_n形状(B,33)；也支持额外时间维。
class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments."""

    def __init__(self, hidden=(2048, 2048, 2048)):
        super().__init__()
        # 输入133维 = 去水平位置/朝向依赖的状态特征71 + 动作29 + 环境参数33。
        self.in_dim = INV_DIM + NUM_J + XI_DIM
        self.net = mlp(self.in_dim, list(hidden), WM_OUT, act=nn.SiLU, layer_norm=True)
        # 初始网络输出为0；反归一化后对应统计均值增量，并非始终零增量。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)

    # 参考目标g不输入动力学网络：不同动作目标共享同一身体的物理规律。
    def inputs(self, S, a, xi_n):
        return torch.cat((invariant_features(S), a, xi_n), -1)

    # collect()用真实增量更新out_norm；8步训练损失实际调用state_error()。
    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    # 【单步预测】一次对应0.02秒控制周期，不调用真实仿真器。
    # 必须保留对状态/动作的梯度，供策略通过本模型做BPTT。
    def step(self, S, a, xi_n):
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n)))
        # MLP预测归一化增量；先还原量纲，再转换回世界坐标并更新状态。
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        # 旋转通过指数映射和矩阵乘法更新，不能直接把3维向量加到旋转矩阵上。
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd)

    # 返回(...,70)的局部状态误差；旋转用相对旋转，不直接减9个矩阵元素。
    # 这里S是真实状态、S_hat是预测；train._wm_err()负责归一化和误差聚合。
    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ Rh),
                (Rt @ (vh - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (wh - w).unsqueeze(-1)).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


# 策略使用观测历史和目标，不直接读取世界模型使用的完整状态。
# 目标编码为32维z，结合最近10帧观测，解码为29维动作。
class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh."""

    def __init__(self, obs_dim: int, goal_dim: int, hist: int = 10, z_dim: int = 32):
        super().__init__()
        self.hist = hist
        self.enc = mlp(goal_dim, [2048, 1024, 512, 256], z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, [2048, 1024, 512, 256], NUM_J)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        o = self.obs_norm(obs_hist).flatten(-2)
        return run_mlp(self.dec, torch.cat((z, o), -1))


# Critic估计想象窗口之后的回报，不预测下一状态。
class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
