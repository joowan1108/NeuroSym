"""DSRL-SAC: soft actor-critic over the latent-noise action space W = [-b_W, b_W]^d.

Follows the pi_0 simulation setup of the DSRL paper (appendix C.5, table 11):
4-layer conv encoders on 64x64 images, MLP actor/critics, 10 critics with mean
aggregation (no clipped double-Q), target entropy -d/2, layer norm (appendix B).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Encoder(nn.Module):
    def __init__(self, in_channels: int, image_size: int, features=(32, 32, 32, 32), out_dim: int = 64):
        super().__init__()
        layers, c = [], in_channels
        for i, f in enumerate(features):
            layers += [nn.Conv2d(c, f, 3, stride=2 if i < 3 else 1), nn.ReLU()]
            c = f
        self.convs = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.convs(torch.zeros(1, in_channels, image_size, image_size)).flatten(1).shape[1]
        self.head = nn.Sequential(nn.Linear(flat, out_dim), nn.LayerNorm(out_dim), nn.Tanh())
        self.out_dim = out_dim

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = img.float() / 255.0 - 0.5
        return self.head(self.convs(x).flatten(1))


class EnsembleLinear(nn.Module):
    def __init__(self, n: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(n, 1, out_dim))
        bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (n, B, in) -> (n, B, out)
        return torch.baddbmm(self.bias, x, self.weight)


class Critic(nn.Module):
    """Q^W(s, w): shared conv encoder + ensemble of MLP heads."""

    def __init__(self, n_critics, in_channels, image_size, prop_dim, act_dim, hidden, n_layers):
        super().__init__()
        self.n = n_critics
        self.encoder = Encoder(in_channels, image_size)
        dims = [self.encoder.out_dim + prop_dim + act_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList(EnsembleLinear(n_critics, a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = EnsembleLinear(n_critics, hidden, 1)

    def forward(self, img, prop, act) -> torch.Tensor:  # (n, B)
        x = torch.cat([self.encoder(img), prop, act], dim=-1)
        x = x.unsqueeze(0).expand(self.n, -1, -1)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(F.layer_norm(x, x.shape[-1:]))
        return self.out(x).squeeze(-1)


class Actor(nn.Module):
    """pi^W(w | s): tanh-Gaussian scaled to the action magnitude b_W."""

    def __init__(self, in_channels, image_size, prop_dim, act_dim, hidden, n_layers, magnitude, init_std=1.0):
        super().__init__()
        self.encoder = Encoder(in_channels, image_size)
        layers, d = [], self.encoder.out_dim + prop_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.magnitude = magnitude
        # Start as (approximately) w ~ N(0, init_std^2), i.e. the base policy's own noise, so
        # that switching from warm-up rollouts to the actor does not change behaviour.
        with torch.no_grad():
            self.mean.weight.mul_(0.01)
            self.mean.bias.zero_()
            self.log_std.weight.mul_(0.01)
            self.log_std.bias.fill_(math.log(_pre_tanh_std(init_std / magnitude)))

    def forward(self, img, prop, deterministic: bool = False):
        h = self.trunk(torch.cat([self.encoder(img), prop], dim=-1))
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        u = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        squashed = torch.tanh(u)
        log_prob = (
            -0.5 * ((u - mean) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        ).sum(-1)
        log_prob = log_prob - torch.log(self.magnitude * (1 - squashed**2) + 1e-6).sum(-1)
        return self.magnitude * squashed, log_prob


def _pre_tanh_std(target: float) -> float:
    """sigma such that tanh(N(0, sigma^2)) has standard deviation `target` (bisection)."""
    u = torch.randn(200_000, generator=torch.Generator().manual_seed(0))
    lo, hi = 1e-3, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if torch.tanh(u * mid).std() < target else (lo, mid)
    return (lo + hi) / 2


class ReplayBuffer:
    def __init__(self, capacity: int, img_shape: tuple, prop_dim: int, act_dim: int):
        self.capacity, self.size, self.ptr = capacity, 0, 0
        self.img = np.zeros((capacity, *img_shape), np.uint8)
        self.next_img = np.zeros((capacity, *img_shape), np.uint8)
        self.prop = np.zeros((capacity, prop_dim), np.float32)
        self.next_prop = np.zeros((capacity, prop_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros((capacity,), np.float32)
        self.done = np.zeros((capacity,), np.float32)

    def add(self, img, prop, act, rew, next_img, next_prop, done) -> None:
        i = self.ptr
        self.img[i], self.prop[i], self.act[i] = img, prop, act
        self.rew[i], self.next_img[i], self.next_prop[i], self.done[i] = rew, next_img, next_prop, done
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        to = lambda a: torch.from_numpy(a[idx]).to(device, non_blocking=True)  # noqa: E731
        return dict(
            img=to(self.img), prop=to(self.prop), act=to(self.act), rew=to(self.rew),
            next_img=to(self.next_img), next_prop=to(self.next_prop), done=to(self.done),
        )


class DSRLAgent:
    def __init__(
        self,
        act_dim: int,
        magnitude: float = 2.0,
        in_channels: int = 9,
        image_size: int = 64,
        prop_dim: int = 16,
        hidden: int = 256,
        n_layers: int = 3,
        n_critics: int = 10,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.999,
        tau: float = 0.005,
        init_alpha: float = 0.01,
        device: str = "cuda",
    ):
        self.cfg = dict(
            act_dim=act_dim, magnitude=magnitude, in_channels=in_channels, image_size=image_size,
            prop_dim=prop_dim, hidden=hidden, n_layers=n_layers, n_critics=n_critics,
            actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, tau=tau, init_alpha=init_alpha,
        )
        self.device = torch.device(device)
        self.gamma, self.tau = gamma, tau
        self.actor = Actor(in_channels, image_size, prop_dim, act_dim, hidden, n_layers, magnitude,
                           init_std=min(1.0, 0.9 * magnitude)).to(self.device)
        make_critic = lambda: Critic(  # noqa: E731
            n_critics, in_channels, image_size, prop_dim, act_dim, hidden, n_layers
        ).to(self.device)
        self.critic, self.critic_target = make_critic(), make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.tensor(math.log(init_alpha), device=self.device, requires_grad=True)
        self.target_entropy = -act_dim / 2
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=critic_lr)
        self.prop_mean = torch.zeros(prop_dim, device=self.device)
        self.prop_std = torch.ones(prop_dim, device=self.device)
        self.grad_steps = 0

    def set_prop_stats(self, prop: np.ndarray) -> None:
        self.prop_mean = torch.as_tensor(prop.mean(0), dtype=torch.float32, device=self.device)
        self.prop_std = torch.as_tensor(prop.std(0) + 1e-3, dtype=torch.float32, device=self.device)

    def _norm(self, prop: torch.Tensor) -> torch.Tensor:
        return ((prop - self.prop_mean) / self.prop_std).clamp(-10, 10)

    @torch.no_grad()
    def act(self, img: np.ndarray, prop: np.ndarray, deterministic: bool = False) -> np.ndarray:
        img_t = torch.from_numpy(img).to(self.device)
        prop_t = self._norm(torch.from_numpy(prop).to(self.device))
        w, _ = self.actor(img_t, prop_t, deterministic)
        return w.cpu().numpy().astype(np.float32)

    def update(self, batch: dict) -> dict:
        prop, next_prop = self._norm(batch["prop"]), self._norm(batch["next_prop"])
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_w, next_logp = self.actor(batch["next_img"], next_prop)
            target_q = self.critic_target(batch["next_img"], next_prop, next_w).mean(0)
            target = batch["rew"] + self.gamma * (1 - batch["done"]) * (target_q - alpha * next_logp)
        q = self.critic(batch["img"], prop, batch["act"])
        critic_loss = ((q - target.unsqueeze(0)) ** 2).mean(1).sum()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        w, logp = self.actor(batch["img"], prop)
        actor_loss = (alpha * logp - self.critic(batch["img"], prop, w).mean(0)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy).mean())
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.lerp_(p, self.tau)
        self.grad_steps += 1
        return dict(
            critic_loss=float(critic_loss), actor_loss=float(actor_loss),
            alpha=float(alpha), q=float(q.mean()),
        )

    def save(self, path) -> None:
        torch.save(
            dict(
                cfg=self.cfg,
                actor=self.actor.state_dict(),
                critic=self.critic.state_dict(),
                critic_target=self.critic_target.state_dict(),
                log_alpha=self.log_alpha.detach().cpu(),
                prop_mean=self.prop_mean.cpu(),
                prop_std=self.prop_std.cpu(),
                grad_steps=self.grad_steps,
                noise_mode=getattr(self, "noise_mode", None),
            ),
            path,
        )

    @classmethod
    def load(cls, path, device: str = "cuda") -> "DSRLAgent":
        ckpt = torch.load(path, map_location="cpu")
        agent = cls(**ckpt["cfg"], device=device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            agent.log_alpha.copy_(ckpt["log_alpha"])
        agent.prop_mean = ckpt["prop_mean"].to(agent.device)
        agent.prop_std = ckpt["prop_std"].to(agent.device)
        agent.grad_steps = ckpt["grad_steps"]
        agent.noise_mode = ckpt.get("noise_mode")
        return agent
