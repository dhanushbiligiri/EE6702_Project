# EE6702_Project

This repository contains a Python implementation of Guided Policy Search (GPS) with iLQR-based guide generation for the Gymnasium MuJoCo `Humanoid-v4` environment.

## Guided Policy Search for MuJoCo Humanoid

This repository contains a Python implementation of Guided Policy Search (GPS) with iLQR-based guide generation for the Gymnasium MuJoCo `Humanoid-v4` environment.

The project includes:

- a Gaussian neural network policy in PyTorch
- iLQR with finite-difference dynamics
- linear-Gaussian guide controllers
- importance-weighted policy optimization
- policy rendering and evaluation utilities

---

### Repository Structure

Typical files in this project:

- `run_pipeline.py`  
  Main training entry point. Builds guides, pretrains the policy, and runs GPS iterations.

- `test_policy.py`  
  Loads a saved policy checkpoint and renders the learned policy in the MuJoCo humanoid environment.

- `gps_loop.py`  
  Main Guided Policy Search training loop.

- `ilqr.py`  
  iLQR / DDP-style trajectory optimization used to generate guide controllers.

- `policy.py`  
  Gaussian policy network.

- `guide.py`  
  Guide trajectory sampling and guide log-probability utilities.

- `importance_objective.py`  
  Importance-weighted GPS policy objective.

- `reward.py`  
  Custom reward used for trajectory optimization and policy learning.

- `env.py`  
  Humanoid environment wrapper.

- `config.py`  
  Central place for reward weights, iLQR settings, GPS settings, and policy settings.

### Steps to implememnt

1. Git clone the repo
2. Install the conda environemnt using -

``conda env create -f environment.yml``

3. To train the policy, run -

``python run_pipeline.py``

By default, the policy checkpoint is saved as:

``best_policy.pt``

4. After training, render the saved policy with:

``python test_policy.py``



## References

1. Levine, S., & Koltun, V. (2013, May). Guided policy search. In International conference on machine learning (pp. 1-9). PMLR.

2. OpenAI. (2025). ChatGPT (GPT-5.3). https://chat.openai.com/ - Used for assistance in code development and debugging.
