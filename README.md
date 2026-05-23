# Adversarial Robustness of Geometric Deep Learning for Protein Stability Prediction

Code for the paper:
> **"Adversarial Robustness of Geometric Deep Learning for Protein Stability Prediction"**
> *Under double-blind review, IEEE DSAA 2026*

---

## Repository Structure

```
adversarial-geostab/
├── attacks/
│   ├── attack_stackelberg.py   # Algorithm 1: Constrained DE attack (leader–follower)
│   ├── attack_differential.py  # Algorithm 2: Differentiable geometric attack
│   └── attack_sama.py          # Algorithm 3: SAMA (State-Aware Meta-Adversary)
├── data/
│   └── validate_s669.py        # S669 zero-leakage validation and geometry utilities
├── scripts/
│   └── train_s8754.py          # Train GeoStab from scratch on S8754
├── pth/                        # Place model checkpoint here (not tracked by git)
├── logs/                       # Trajectory plots saved here (not tracked by git)
├── results/                    # CSV outputs saved here (not tracked by git)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup

### 1. Clone this repo and GeoStab side by side

All scripts resolve paths relative to the project root automatically — no manual path editing required.

```bash
git clone https://github.com/YOUR_USERNAME/adversarial-geostab.git
cd adversarial-geostab

# GeoStab must sit one level up, as a sibling directory
cd ..
git clone https://github.com/Gonglab-THU/GeoStab.git
```

Your directory layout should look like this:

```
parent/
├── adversarial-geostab/   ← this repo
└── GeoStab/               ← cloned here
    └── model_ddG_3D/
        └── model.pt
```

### 2. Install dependencies

```bash
cd adversarial-geostab
pip install -r requirements.txt
```

### 3. Place your trained checkpoint

```bash
cp /path/to/best_model_s8754.pth adversarial-geostab/pth/
```

### 4. Prepare S669 evaluation features

```bash
python data/validate_s669.py
```

---

## Training

Train GeoStab from scratch on S8754:

```bash
python scripts/train_s8754.py
```

The best checkpoint (lowest validation MSE) is saved to `pth/best_model_s8754.pth`.

---

## Running Attacks

All scripts support `--sample <mut_id>` for a single mutation or `--num-samples N` for benchmark mode.

### Algorithm 1 — Stackelberg DE Attack

```bash
# Single sample (full settings)
python attacks/attack_stackelberg.py --sample mut_0 --max-iter 100 --pop-size 30

# Benchmark (fast)
python attacks/attack_stackelberg.py --num-samples 30 --max-iter 3 --pop-size 2
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample` | — | Single mutation ID |
| `--num-samples` | 30 | Benchmark sample count |
| `--max-iter` | 3 | DE max iterations |
| `--pop-size` | 2 | DE population size |

### Algorithm 2 — Differentiable Geometric Attack

```bash
python attacks/attack_differential.py --sample mut_0 --steps 50 --lr-pos 0.05 --lr-emb 0.05

python attacks/attack_differential.py --num-samples 30 --steps 50
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--steps` | 50 | Gradient ascent steps |
| `--lr-pos` | 0.05 | Coordinate learning rate |
| `--lr-emb` | 0.05 | Embedding learning rate |
| `--reg` | 0.05 | Regularization strength |
| `--k-rad` | 10 | Residue neighborhood size |

### Algorithm 3 — SAMA

```bash
# Single sample — saves trajectory plot to logs/
python attacks/attack_sama.py --sample mut_0 --episodes 5 --steps 10

# Full S669 benchmark
python attacks/attack_sama.py --num-samples all --episodes 5 --steps 10
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 5 | SAMA episodes |
| `--steps` | 10 | Steps per episode |
| `--reg` | 0.05 | Regularization penalty |
| `--k-rad` | 10 | Residue neighborhood size |
| `--lr-pos` | 0.05 | Coordinate learning rate |
| `--lr-emb` | 0.05 | Embedding learning rate |

---

## Key Results

| Attack | Mean Drift (kcal/mol) | Mean RMSD (Å) |
|--------|-----------------------|----------------|
| Random baseline | 0.034 | ≤ 0.30 |
| FGSM | 1.779 | ≤ 0.30 |
| Stackelberg DE | 0.683 | 0.259 |
| Differentiable | — | ≤ 0.30 |
| SAMA | 1.993 | ≤ 0.30 |
| PGD-10step | 8.353 | ≤ 0.30 |

ESM-2 embeddings account for ~98.9% of adversarial sensitivity.
TRADES and PGD-AT both reduce adversarial drift significantly (p < 0.001).

---

## Citation

```bibtex
@inproceedings{adversarial_geostab_2026,
  title     = {Adversarial Robustness of Geometric Deep Learning
               for Protein Stability Prediction},
  booktitle = {Proceedings of IEEE DSAA 2026},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
