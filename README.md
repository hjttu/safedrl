# G-MATrans-Lagr Reproduction

This is a compact PyTorch reproduction of the paper algorithm in
`25_IFAC_G_MAPPO_Lagr.pdf`: **Primal-Dual based Safe Multi-Agent
Reinforcement Learning with Graph Information Aggregation**.

Implemented pieces:

- Multi-UAV navigation CMDP with second-order integrator dynamics.
- Local graph observations with agent, target, and obstacle nodes.
- Transformer-based graph encoder for variable-size local observations.
- MAPPO-style clipped policy update with centralized reward and cost critics.
- Primal-dual Lagrangian multiplier update for safety-cost constraints.

## Install

```powershell
pip install -r requirements.txt
```

## Train

```powershell
python scripts/train_gmatrans_lagr.py --agents 3 --updates 200 --steps 128
```

For larger experiments matching the paper, run with `--agents 6` or `--agents 9`.
The paper used much longer training on GPU hardware, so the defaults here are
kept small enough for smoke testing.

## Paper-to-code Map

- Eq. (4), UAV dynamics: `gmarlagr/env.py`
- Graph construction and node/edge features: `MultiUAVNavEnv.observe`
- Transformer graph aggregation: `GraphTransformerEncoder`
- Hybrid advantage, Eq. (6): `GMATransLagrTrainer.update`
- Lagrangian multiplier update, Eq. (7)-(9): `_update_lagrange`
- PPO objective and reward/cost critic losses, Eq. (10)-(12): `update`

## Notes

The paper mentions extra training configurations in the authors' repository but
does not list all hyperparameters in the PDF. This repo therefore aims to be a
faithful algorithmic reproduction rather than a bitwise reproduction of their
reported curves.
