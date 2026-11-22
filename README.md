# ABC-EM: Simple Tag

Reference implementation of *Active Behavioral Cloning via Entropy Maximization in
Competitive Environments* on MPE2 `simple_tag_v3`. ABC-EM learns an evader that
imitates a frozen PPO pursuer through online behavioral clones, rewarded for
reaching states where the pursuer's policy is uncertain and its clone is still
changing.

Python 3.11 or 3.12. MIT licensed.

## Install

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```sh
python run.py --output outputs
python plot.py --output outputs
```

`--smoke` runs a short execution check, `--phase pretrain|train` splits the two
stages, and `--device cuda` selects a CUDA build. Default execution uses CPU with
one PyTorch thread.

## Environment

One pursuer (`adversary_0`), one evader (`agent_0`), two obstacles, five discrete
actions, 100 decision cycles per episode. Decisions alternate: the pursuer acts,
the environment advances, and the evader acts on the updated state. Observations
hold 12 values for the pursuer and 10 for the evader. A PPO pursuer is pretrained
for 500,000 transitions against uniformly sampled evader actions and then frozen;
its actions are sampled from the categorical policy.

## Method

A run collects 100,000 pursuer state/action pairs. Every 1,000 interactions it
collects with the current evader, fits a clone and selects it by held-out NLL,
refreshes visitation counts and relabels the replay, then trains the evader
controller against the frozen reward. With `p` the current clone, `p_prev` the
preceding one, and `N(s)` the visitation count of `s`:

```text
beta(s) = 1 - exp(-delta * KL(p(.|s) || p_prev(.|s)))
raw(s)  = H(p(.|s)) * beta(s) + warmup_b / (N(s) + 1)
intrinsic(s) = reward_scale * tanh(tanh(raw(s) / reward_scale) / rms / reward_scale)
controller_reward = intrinsic(next_pursuer_state) - boundary_cost
```

Defaults: delta=30, warmup_b=1.5, reward_scale=10, joint bin width 0.5. The clone
uses 2x128 ReLU layers with AdamW, lr 3e-4, batch 256, and up to 5,000 updates.
The evader uses Double DQN with 2x128 ReLU layers, Adam, lr 3e-4, batch 512,
replay 20,000, target sync every 250 updates, and epsilon 1 to 0.05 over 10,000
interactions.

## Files

`configs/default.json` accepts `config` and `stages` objects; omitted fields use
the defaults in `simple_tag/config.py` and `simple_tag/staged.py`. Each seed writes
`metrics.csv`, `inner_curves.csv`, `rounds.json`, `latest.pt`, `trajectories.npz`,
and an artifact hash manifest. `plot.py` draws NLL, KL, and coverage curves with
editable CSV data and PNG/PDF figures.
