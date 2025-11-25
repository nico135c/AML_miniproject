# AML Mini Project – Steel Plate Fault Classification

This project implements and compares two machine learning classifiers on the **UCI Steel Plates Faults** dataset:

- **Binary Classifier** – predicts whether a fault is *surface-related* or *structural*
- **Multi-Class Classifier** – predicts the *specific fault type* (7 classes)

Both models are trained and evaluated in a reproducible pipeline that generates:

- Confusion matrices
- Classification reports

---

## Dataset

The project uses the **Steel Plates Faults** dataset from the UCI Machine Learning Repository.

- Input: geometric and physical measurements of steel plates
- Output labels: 7 fault indicators (one-hot encoded)

The dataset is automatically downloaded via:

```python
from ucimlrepo import fetch_ucirepo
```

---

## Models

### Binary Classifier

Goal: classify faults as:

- `0` – Structural faults
- `1` – Surface faults

Model: **Logistic Regression**

Label mapping:

| Category | Labels |
|----------|--------|
| Structural (0) | Pastry, Bumps, Other_Faults |
| Surface (1) | Z_Scratch, K_Scratch, Stains, Dirtiness |

---

### Multi-Class Classifier

Goal: predict one of **7 fault types**:

```
Bumps
Dirtiness
K_Scratch
Other_Faults
Pastry
Stains
Z_Scratch
```

Model: **Random Forest Classifier**

---

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/nico135c/AML_miniproject.git
cd AML_miniproject
```

### 2. Install dependencies

```bash
pip install ucimlrepo pandas scikit-learn matplotlib
```

### 3. Run the project

```bash
python main.py
```

This will:

Fownload the dataset  
Train both classifiers  
Generate evaluation figures in `figures/`

---
