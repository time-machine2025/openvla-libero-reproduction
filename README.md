# OpenVLA-LIBERO Reproduction Scaffold

This repository is a small-scale, self-contained reproduction scaffold inspired by the OpenVLA and LIBERO workflow.

It is intentionally honest:

- it does **not** claim to reproduce official OpenVLA or LIBERO numbers
- it **does** show the full research loop: dataset loading, policy fitting, held-out evaluation, and result logging

The goal is to present a reviewable codebase that demonstrates the kind of engineering work you would do before scaling to the real benchmark stack.

## What A Reviewer Sees Quickly

- a language-conditioned manipulation dataset in JSONL format
- a tiny environment with pick-place dynamics
- a behavior-cloning style baseline
- heuristic and random baselines for sanity checking
- saved metrics in `results/latest_metrics.json`

## What Is Included

- a tiny grid-based manipulation environment
- JSONL episodes that look like language-conditioned imitation data
- a simple behavior-cloning style policy learned from demonstrations
- a heuristic fallback policy
- a random baseline
- an evaluation script with JSON output

## Project Layout

```text
openvla-libero-reproduction/
├── configs/
├── data/
├── results/
├── scripts/
├── src/openvla_libero_reproduction/
└── tests/
```

## Quick Start

```bash
python3 scripts/run_reproduction.py
```

Requirements:

- Python 3.10+
- no third-party dependencies for the current toy scaffold

That command will:

1. load the training demonstrations
2. fit the imitation policy
3. generate held-out evaluation tasks
4. compare learned, heuristic, and random policies
5. save metrics to `results/latest_metrics.json`

Current sample output:

```text
behavior_cloning | success_rate=1.00 | avg_steps=8.17
heuristic        | success_rate=1.00 | avg_steps=8.17
random           | success_rate=0.00 | avg_steps=20.00
```

## Research Framing

This repository is meant to read like a good first-stage research scaffold:

- start from a clean task abstraction
- separate training and evaluation data
- compare against non-learning baselines
- save machine-readable results instead of screenshots only

## Minimal Verification

```bash
python3 -m unittest discover -s tests -q
```


## Future Extensions

- swap the synthetic JSONL data for real `lerobot` or `rlds` format
- add vision observations and offline precomputed features
- replace the tabular policy with a transformer or action chunking policy
- integrate official LIBERO environments when compute and dependencies are available
