from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from policy import GaussianPolicy


def pretrain_policy_from_guides(
    policy: GaussianPolicy,
    guide_obs: np.ndarray,
    guide_actions: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float = 1e-3,
    device: str = "cpu",
    show_progress: bool = True,
):
    policy.to(device)
    policy.train()

    obs_t = torch.as_tensor(guide_obs, dtype=torch.float32)
    act_t = torch.as_tensor(guide_actions, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(obs_t, act_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    epoch_iter = range(epochs)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Policy Pretrain", leave=False)

    for epoch in epoch_iter:
        epoch_losses = []

        batch_iter = loader
        if show_progress:
            batch_iter = tqdm(loader, desc=f"Pretrain epoch {epoch+1}/{epochs}", leave=False)

        for obs_b, act_b in batch_iter:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)

            logp = policy.log_prob_actions(obs_b, act_b)
            loss = -logp.mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()

            epoch_losses.append(float(loss.item()))

        if show_progress and len(epoch_losses) > 0:
            tqdm.write(
                f"[Pretrain] epoch={epoch+1}/{epochs} | mean_loss={np.mean(epoch_losses):.6f}"
            )