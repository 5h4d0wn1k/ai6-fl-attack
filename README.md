> **⚠️ EDUCATIONAL USE ONLY — AUTHORIZED TESTING ONLY.**
> This project exists for education, research, and **defense of systems you own
> or hold explicit written authorization to assess**. Unauthorized use is
> prohibited and may be illegal. Read [ETHICS.md](ETHICS.md) and
> [SCOPE.md](SCOPE.md) before use. Use at your own risk; **AS IS**, no warranty.

# AI6 — Federated Learning Attack & Defense Framework

Offline security playground for **federated learning**: model-update poisoning,
gradient inversion, free-rider attacks, and Byzantine-robust aggregation
defenses for **artificial-intelligence security** research.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/5h4d0wn1k/ai6-fl-attack)](https://github.com/5h4d0wn1k/ai6-fl-attack)
[![Issues](https://img.shields.io/github/issues/5h4d0wn1k/ai6-fl-attack)](https://github.com/5h4d0wn1k/ai6-fl-attack/issues)
[![Last commit](https://img.shields.io/github/last-commit/5h4d0wn1k/ai6-fl-attack)](https://github.com/5h4d0wn1k/ai6-fl-attack)

## Why

Federated learning is attractive exactly because training data never leaves the
client — but the model *updates* do, and those updates are an attack surface.
A handful of malicious clients can poison aggregation with flipped gradients,
free-ride on honest work, or even invert gradients to reconstruct private
training samples. Defenders need a controllable, offline environment to measure
how far an attack degrades accuracy and how well robust aggregation (Krum,
trimmed mean) recovers it. AI6 simulates a complete federation — clients,
server, aggregators, and diver — entirely on your own machine with only `numpy`
as a dependency, so every attack and defense code path can be exercised, scored,
and documented without touching any real ML infrastructure.

## Features

- **Full federation simulation** — `FederatedClient` local training + `FederatedServer` (mean, Krum, trimmed-mean aggregation)
- **Model-update poisoning** — sign-flip and scale poisoning (`ModelUpdatePoisoner`)
- **Gradient inversion** — single and batch sample recovery (`GradientInverter`)
- **Free-rider attack** — fake-update generation with honest-vs-free-rider benefit metric
- **Byzantine detection** — outlier detection over a 6-client pool (`ByzantineDetector`)
- **Deterministic offline runs** — `--clients`, `--rounds`, `--features`, `--seed`
- **JSON reports** — `--output reports/ai6-report.json`, plus `--quiet` CI mode

## Quickstart

Only dependency: `numpy`.

```bash
# Offline demo — baseline, poisoning, inversion, free-rider, Krum, trimmed-mean (exit 0)
python3 fl_attack.py

# Tunable experiment
python3 fl_attack.py --clients 6 --rounds 5 --features 10 --seed 42

# JSON report
python3 fl_attack.py --output reports/ai6-report.json

# Quiet CI mode + JSON
python3 fl_attack.py --quiet --output reports/ai6-report.json

# Unit tests (7 cases)
python3 -m unittest discover -s tests -v
```

Exit codes: `0` clean experiment, `1` bad arguments / report write failure.

## Example output

```
[1] Baseline federated learning (no attacks)...      accuracy 0.7800
[2] Model update poisoning (sign flip)...            accuracy 0.3200, drop -0.4600
[3] Gradient inversion attack...                     recovered norm, shape (1, 10)
[4] Free-rider attack...                             benefit True
[5] Byzantine bypass (Krum)...                       0.7200 (+0.4000 over poisoned)
[6] Trimmed mean defense...                          0.7400
```

## Project structure

- `fl_attack.py` — CLI entry point and full attack/defense simulation
- `tests/` — 7 unit tests (client shapes, mean/Krum/trimmed math, Byzantine detection, determinism, JSON write)

## Legal & authorized use

For **educational and authorized security research** only. Run against simulated
federations you own or lab systems you are explicitly authorized to assess. Do
not invert gradients from real production models or live federation pools. See
[ETHICS.md](ETHICS.md), [SCOPE.md](SCOPE.md), and [SECURITY.md](SECURITY.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).