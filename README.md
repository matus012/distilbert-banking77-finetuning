# Banking77 Intent Classification: Fine-Tuning DistilBERT and BERT for 77-Class Banking Query Classification

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange)

TUKE FEI, BSc Intelligent Systems Y2, ML Course Assignment 20 (Zadanie 20 — Ladenie Transformer).

## Key Results

| Run | Model | LR | Epochs | Freeze | Test Macro-F1 | Wall-clock |
|-----|-------|----|--------|--------|---------------|------------|
| distilbert_epochs5 | DistilBERT-base | 5e-5 | 5 | none | **0.9307** | 188s |
| bert_baseline | BERT-base | 5e-5 | 3 | none | 0.9289 | 197s |
| distilbert_lr_high | DistilBERT-base | 1e-4 | 3 | none | 0.9273 | 113s |

**Best result:** `distilbert_epochs5` — test macro-F1 **0.9307**, accuracy **0.9305**, 188s on RTX 4060 Laptop.
DistilBERT with 5 epochs beats BERT-base (3 epochs) at roughly half the wall-clock cost.

![Test Macro-F1 per run](experiments/_summary/plots/test_macro_f1_bar.png)

Full report: [REPORT.md](REPORT.md)

## Repository Structure

```
transformer_project/
├── src/
│   ├── config.py          # ExperimentConfig dataclass + 8 factory functions
│   ├── data.py            # Banking77 loader + tokenizer
│   ├── train.py           # HF Trainer wrapper (freeze strategies, CSV logging)
│   ├── evaluate.py        # Test-set evaluation + plots
│   ├── analysis.py        # Cross-run aggregation helpers
│   └── app.py             # Gradio demo app
├── scripts/
│   ├── run_all_experiments.py  # Sequential sweep runner
│   └── ...
├── notebooks/
│   └── analysis.ipynb     # Cross-run comparison notebook
├── experiments/
│   ├── distilbert_baseline/    # Run artifacts (config, metrics, plots)
│   ├── distilbert_epochs5/     # ... (best model)
│   ├── ...
│   └── _summary/               # Aggregated cross-run plots + summary table
├── archive/
│   ├── decisions.md       # Design decisions log
│   └── debug_log.md       # Issue log
├── context.md             # Project plan (static)
├── status.txt             # Current phase + per-phase summary
├── run_app.bat            # Windows one-click launcher
├── requirements.txt
└── pyproject.toml
```

## Quick Start

```bash
git clone https://github.com/matus012/transformer_project.git
cd transformer_project
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
pip install -e .
```

> **Note:** `experiments/*/best_model/` is gitignored — model weights are not included in the repo.
> You must train before running the demo. Two options:
>
> **Option A — Single best run** (~188s on RTX 4060 + fp16):
> ```bash
> python scripts/run_all_experiments.py --only distilbert_epochs5
> ```
>
> **Option B — Full 8-run sweep** (~17 min on RTX 4060 + fp16):
> ```bash
> python scripts/run_all_experiments.py
> ```

## Reproduce Experiments

**Single baseline run:**
```bash
python -m src.train --config baseline
```

**Full 7-run sweep (runs 2-8):**
```bash
python scripts/run_all_experiments.py
```

**Single named run:**
```bash
python scripts/run_all_experiments.py --only distilbert_epochs5
```

**Evaluate a saved run on test set:**
```bash
python -m src.evaluate --run_dir experiments/distilbert_epochs5 --split test
```

**Cross-run analysis notebook:**
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Run the Demo

After training `distilbert_epochs5` (see Quick Start above):

**Windows (one-click):**
```
run_app.bat
```

**Manual:**
```bash
python src/app.py
```

Open http://127.0.0.1:7860 in a browser. Enter any banking query to predict the intent with top-3 confidence scores.

## Hardware

RTX 4060 Laptop 8GB VRAM, i7-13650HX, 16GB RAM. All training uses fp16. Total GPU time for all 8 runs: ~20 minutes.

## License

[MIT](LICENSE)
