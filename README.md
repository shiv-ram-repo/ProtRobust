# Adversarial Robustness of Geometric Protein Stability Predictors under Physically Constrained Perturbations

Official implementation for the paper:

> **Adversarial Robustness of Geometric Protein Stability Predictors under Physically Constrained Perturbations**
>
> Accepted for presentation and publication at **IEEE DSAA 2026 (Long Presentation)**.

## Repository Structure

```text
ProtRobust/
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

## Setup

### 1. Clone this repository and GeoStab side by side

All scripts resolve paths relative to the project root automatically, so no manual path editing is required.

```bash
git clone git@github.com:shiv-ram-repo/ProtRobust.git
cd ProtRobust

# GeoStab must sit one level up, as a sibling directory
cd ..
git clone https://github.com/Gonglab-THU/GeoStab.git
```

Your directory layout should look like:

```text
parent/
├── ProtRobust/                ← this repository
└── GeoStab/                   ← cloned here
    └── model_ddG_3D/
        └── model.pt
```

### 2. Install dependencies

```bash
cd ProtRobust
pip install -r requirements.txt
```

### 3. Place your trained checkpoint

Place the trained GeoStab checkpoint in the `pth/` directory:

```bash
cp /path/to/best_model_s8754.pth pth/
```

The expected path is:

```text
ProtRobust/pth/best_model_s8754.pth
```

### 4. Prepare S669 evaluation features

```bash
python data/validate_s669.py
```

## Training

Train GeoStab from scratch on S8754:

```bash
python scripts/train_s8754.py
```

The best checkpoint, selected by the lowest validation MSE, is saved to:

```text
pth/best_model_s8754.pth
```

## Running Attacks

All attack scripts support single-sample evaluation using `--sample <mut_id>` and benchmark evaluation using `--num-samples N`.

### Algorithm 1 — Stackelberg DE Attack

The constrained differential-evolution attack uses attention-guided residue selection under the physical RMSD constraint.

#### Single sample

```bash
python attacks/attack_stackelberg.py \
    --sample mut_0 \
    --max-iter 100 \
    --pop-size 30
```

#### Benchmark mode

```bash
python attacks/attack_stackelberg.py \
    --num-samples 30 \
    --max-iter 3 \
    --pop-size 2
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample` | — | Single mutation ID |
| `--num-samples` | 30 | Number of benchmark samples |
| `--max-iter` | 3 | Maximum DE iterations |
| `--pop-size` | 2 | DE population size |

### Algorithm 2 — Differentiable Geometric Attack

This attack enables gradient-based optimization through the geometric feature extraction pipeline.

#### Single sample

```bash
python attacks/attack_differential.py \
    --sample mut_0 \
    --steps 50 \
    --lr-pos 0.05 \
    --lr-emb 0.05
```

#### Benchmark mode

```bash
python attacks/attack_differential.py \
    --num-samples 30 \
    --steps 50
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--steps` | 50 | Gradient-ascent steps |
| `--lr-pos` | 0.05 | Coordinate learning rate |
| `--lr-emb` | 0.05 | Embedding learning rate |
| `--reg` | 0.05 | Regularization strength |
| `--k-rad` | 10 | Residue neighborhood size |

### Algorithm 3 — SAMA

SAMA (State-Aware Meta-Adversary) combines gradient-based optimization with sensitivity-weighted stochastic exploration.

#### Single sample

The following command also saves the optimization trajectory to `logs/`:

```bash
python attacks/attack_sama.py \
    --sample mut_0 \
    --episodes 5 \
    --steps 10
```

#### Full S669 benchmark

```bash
python attacks/attack_sama.py \
    --num-samples all \
    --episodes 5 \
    --steps 10
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 5 | Number of SAMA episodes |
| `--steps` | 10 | Steps per episode |
| `--reg` | 0.05 | Regularization penalty |
| `--k-rad` | 10 | Residue neighborhood size |
| `--lr-pos` | 0.05 | Coordinate learning rate |
| `--lr-emb` | 0.05 | Embedding learning rate |

## Key Results

The main results reported in the paper are summarized below.

| Attack | Mean Drift (kcal/mol) | Mean RMSD (Å) |
|--------|------------------------|---------------|
| Random baseline | 0.034 | ≤ 0.30 |
| FGSM | 1.779 | ≤ 0.30 |
| Stackelberg DE | 0.683 | 0.259 |
| Differentiable | — | ≤ 0.30 |
| SAMA | 1.993 | ≤ 0.30 |
| PGD-10step | 8.353 | ≤ 0.30 |

The factorial analysis attributes approximately **98.9% of measured adversarial sensitivity to ESM-2 sequence embeddings**.

TRADES and PGD adversarial training both reduce adversarial drift significantly while maintaining negligible clean-performance degradation (`p < 0.001`).

## Reproducibility

The experiments use the following primary benchmarks:

- **S8754**: training benchmark
- **S669**: zero-leakage evaluation benchmark

The S669 evaluation is constructed to avoid sequence overlap with proteins in S8754.

Coordinate perturbations are evaluated under the physically constrained RMSD setting described in the paper.

## Citation

If you use this code or the results in your work, please cite:

```bibtex
@inproceedings{shivram2026adversarial,
  author    = {A. Shivram and Tanmay Kumar Dalai and Aneesh Sreevallabh Chivukula and Manik Gupta},
  title     = {Adversarial Robustness of Geometric Protein Stability Predictors under Physically Constrained Perturbations},
  booktitle = {Proceedings of IEEE DSAA 2026},
  year      = {2026}
}
```

## Authors

**A. Shivram**, **Tanmay Kumar Dalai**, **Aneesh Sreevallabh Chivukula**, and **Manik Gupta**

Birla Institute of Technology and Science, Pilani, Hyderabad Campus, India.

## License

MIT License. See [LICENSE](LICENSE) for details.
