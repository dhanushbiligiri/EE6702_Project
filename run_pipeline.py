from __future__ import annotations
import torch

from config import GPSConfig
from gps_loop import GuidedPolicySearchTrainer
from evaluate import evaluate_policy


def main():
    cfg = GPSConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = GuidedPolicySearchTrainer(cfg, device=device)
    result = trainer.train()


    policy = result["policy"]
    torch.save(policy.state_dict(), "best_policy.pt")
    print("Saved best policy to best_policy.pt")

    env = trainer.env
    init_state = env.reset(seed=cfg.seed)

    eval_out = evaluate_policy(
        env=env,
        policy=policy,
        init_state=init_state,
        horizon=cfg.ilqr.horizon,
        device=trainer.device,
        n_rollouts=5,
        deterministic=True,
    )

    print("\nFinal evaluation:")
    print(eval_out)


if __name__ == "__main__":
    main()