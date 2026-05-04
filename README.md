# Predicting CYP2C19 Inhibition: A Machine Learning Approach

A Jupyter notebook implementing a machine learning pipeline to predict whether a molecule inhibits the **CYP2C19** enzyme — a major cytochrome P450 isoform responsible for metabolising approximately 75% of marketed drugs. Accurate inhibition prediction helps prevent adverse drug-drug interactions early in the drug discovery process.

**Authors:** Alessio Carnevale, Manuel Cattoni, Carlo Schillaci

---

## Overview

The notebook evaluates four ML algorithms across three distinct molecular feature representations, integrating model explainability (SHAP) and an Applicability Domain (AD) assessment to produce a deployment-ready QSAR classifier.

**Best result:** SVM (RBF) on combined descriptors + ECFP4 fingerprints — **Test AUC: 0.8878, F1: 0.8091**

---

## Dataset

- **Source:** Veith et al. CYP2C19 high-throughput screening dataset, accessed via the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/)
- **Size:** 12,665 unique molecules (5,819 inhibitors / 6,846 non-inhibitors) after SMILES validation and deduplication
- **Split:** Pre-computed scaffold-based split from TDC (70% train / 10% validation / 20% test), preserving the ~46/54% class balance across all sets

---

## Notebook Structure

| Section | Description |
|---|---|
| **Data Cleaning & Scaffold Preservation** | SMILES validation via RDKit, canonicalisation, deduplication, and cross-split leakage removal |
| **1. Exploratory Data Analysis (EDA)** | Class balance, SMILES length distributions, descriptor distributions, Spearman correlation matrix, sample molecule visualisations |
| **References: Descriptor Justification** | Literature grounding for each descriptor choice (Lipinski, Veber, Lovering, Balaban, Bertz, etc.) |
| **Feature Engineering** | Computation of 19 physicochemical descriptors → pruned to 17 after multicollinearity filtering (threshold ρ > 0.85); ECFP4 Morgan fingerprints (radius=2, 1024 bits) |
| **2. Feature Subset Experiment** | Mutual Information–based `SelectKBest` evaluation across 20/50/80/100% feature retention |
| **3. Model Implementation & Comparison** | Training of Logistic Regression, Random Forest, XGBoost, and SVM (RBF) across all three feature sets (Descriptors / Morgan FP / Combined); validation-based hyperparameter tuning |
| **3.1 Dimensionality Reduction** | TruncatedSVD (1024 → 256 dims) on fingerprints; K-Means clustering (35 centroids) robustness check on descriptors |
| **4. Model Evaluation & Visualisations** | ROC-AUC and PR-AUC curves, full performance table across all model/feature combinations |
| **5. Model Explainability & Reliability** | SHAP beeswarm and waterfall plots (XGBoost proxy); Applicability Domain via Williams Plot; Tanimoto similarity threshold analysis |

---

## Feature Representations

### Physicochemical Descriptors (17D)
A curated set selected for chemical relevance to the CYP2C19 binding pocket:

| Category | Features |
|---|---|
| Pharmacokinetic baseline | `logp`, `mol_wt`, `hbond_donor`, `hbond_acceptor`, `tpsa`, `num_rotatable_bonds` |
| Structural fragments | `num_aromatic_rings`, `fr_Al_OH`, `fr_NH2`, `fr_COO2` |
| Electronic properties | `max_partial_charge`, `min_partial_charge`, `num_heteroatoms` |
| Halogenation | `fr_halogen` |
| 3D shape proxies (2D) | `fraction_csp3`, `labute_asa` |
| Topological complexity | `balaban_j`, `bertz_ct` |
| Polarizability | `mol_mr` |

### Structural Fingerprints
- **ECFP4** (Morgan radius=2, 1024 bits) — industry-standard structural benchmark

### Combined
- Concatenation of the 17D descriptor set and ECFP4 fingerprints (1041 features total)

---

## Models & Results

| Feature Set | Model | Test AUC | Test F1 |
|---|---|---|---|
| **Combined** | **SVM (RBF)** | **0.8878** | **0.8091** |
| Morgan FP | SVM (RBF) | 0.8391 | 0.7338 |
| Descriptors | XGBoost | 0.8662 | 0.7873 |
| Descriptors | SVM (RBF) | 0.8509 | 0.7735 |
| Morgan FP | Random Forest | 0.8251 | 0.7332 |

Key findings:
- The 17D physicochemical descriptor set **outperformed** the much larger 1024-bit fingerprint-only representation across top models.
- Combining both representations consistently outperformed either alone.
- Tree-based models (RF, XGBoost) exhibited significant overfitting (~98–99% train accuracy vs. ~79–80% test), while SVM showed more controlled generalisation.
- Dimensionality reduction via TruncatedSVD and K-Means did not substantially degrade performance.

---

## Explainability & Deployment Readiness

- **SHAP** (Shapley Additive Explanations): Beeswarm and waterfall plots generated on the best descriptor-only model (XGBoost) to extract chemically interpretable feature attributions. Top drivers: `logp`, `fraction_csp3`, `bertz_ct`.
- **Applicability Domain**: Williams Plot methodology using leverage thresholds on the physicochemical descriptor matrix. Over **96.0%** of test molecules fall within the safe interpolation boundary. A Tanimoto similarity check confirms **96.5%** of test molecules share ≥ 0.3 similarity with the training set.

---

## Requirements

```
# Core
numpy
pandas
matplotlib
seaborn

# Machine Learning
scikit-learn
xgboost
shap

# Cheminformatics
rdkit

# Dataset
PyTDC
```

Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap rdkit PyTDC
```

---

## Usage

1. Clone the repository and install dependencies.
2. Open `notebook.ipynb` in JupyterLab or Jupyter Notebook.
3. Run all cells sequentially — the TDC dataset is downloaded automatically on first run.

> **Note:** The scaffold-based split is fetched directly from TDC and preserved throughout. Do not re-split the data, as this would break the structural novelty guarantee of the evaluation.

---

## References

1. Veith, H., et al. (2009). *Nature Biotechnology*, 27(11), 1050–1055.
2. Lewis, D. F. V., et al. (2007). *Toxicology in Vitro*, 21(4), 136–142.
3. Lipinski, C. A., et al. (1997). *Advanced Drug Delivery Reviews*, 23(1-3), 3–25.
4. Veber, D. F., et al. (2002). *Journal of Medicinal Chemistry*, 45(12), 2615–2623.
5. Lovering, F., et al. (2009). *Journal of Medicinal Chemistry*, 52(21), 6752–6756.
6. Balaban, A. T. (1982). *Chemical Physics Letters*, 89(5), 399–404.
7. Bertz, S. H. (1981). *Journal of the American Chemical Society*, 103(12), 3599–3601.
