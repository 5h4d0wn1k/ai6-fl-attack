# AI6 — Federated Learning Attack

Demonstrates attacks and defenses in federated learning systems.

## Overview

This project implements various attacks against federated learning:
- **Model update poisoning**: Corrupting local updates before aggregation
- **Gradient inversion**: Recovering training data from model updates
- **Free-rider attack**: Benefiting from aggregation without contributing
- **Byzantine tolerance bypass**: Circumventing robust aggregation methods

## Features

- Complete federated learning simulation (clients + server)
- Multiple aggregation strategies (mean, Krum, trimmed mean)
- Attack implementations (sign flip, scale, Gaussian noise)
- Gradient inversion via optimization
- Byzantine detection and filtering

## Installation

```bash
# No external dependencies - numpy only
python3 -c "import numpy; print('numpy available')"
```

## Usage

```bash
python3 fl_attack.py
```

## Example Output

```
============================================================
AI6 — Federated Learning Attack Demonstration
============================================================

[1] Baseline federated learning (no attacks)...
    Baseline accuracy after 5 rounds: 0.7800

[2] Model update poisoning (sign flip attack)...
    Accuracy under poisoning: 0.3200
    Accuracy drop: -0.4600

[3] Gradient inversion attack...
    Recovered gradient norm: 2.3456
    Inverted sample shape:  (1, 10)
    Batch inversion shape:  (3, 10)

[4] Free-rider attack...
    Honest client accuracy: 0.7100
    Free-rider accuracy:    0.6900
    Free-rider got benefit: True

[5] Byzantine tolerance bypass (Krum aggregation)...
    Krum accuracy (with detection): 0.7200
    vs poisoned mean: 0.3200
    Improvement: +0.4000

[6] Trimmed mean aggregation defense...
    Trimmed mean accuracy: 0.7400

============================================================
Results Summary:
  Baseline:          0.7800
  Under poisoning:   0.3200
  Krum defense:      0.7200
  Trimmed defense:   0.7400
  Free-rider benefit: 0.6900
============================================================
```

## Legal Disclaimer

**IMPORTANT: Read before use.**

This project is provided for **educational and authorized security testing purposes only**.

### Authorization Requirements
- You MUST have explicit written permission from the system owner before using this tool
- Unauthorized manipulation of federated learning systems may violate privacy laws
- This tool should ONLY be used on systems you own or have written authorization to test

### Legal Framework
- **Computer Fraud and Abuse Act (CFAA)**: Unauthorized access to computer systems is a federal crime
- **Privacy Regulations**: Federated learning often involves sensitive data
- **AI Security Regulations**: Emerging regulations may govern AI/ML system manipulation
- **Data Protection Laws**: Gradient inversion may reveal private training data

### Acceptable Use
- Testing security of your own federated learning systems
- Authorized red team exercises with written scope
- Academic research in controlled lab environments
- Security education and training

### Prohibited Use
- Attacking federated learning systems without authorization
- Inverting gradients to steal private training data
- Any activity that violates applicable laws or regulations
- Commercial use without proper licensing

### No Warranty
This software is provided "AS IS" without warranty of any kind. The author is not responsible for any misuse or damage caused by this software.

### Responsible Disclosure
If you discover vulnerabilities using this tool, follow responsible disclosure practices:
1. Report to the vendor/owner privately
2. Allow reasonable time for remediation
3. Do not exploit beyond proof of concept

## License

MIT
