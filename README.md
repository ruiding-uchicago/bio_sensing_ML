# Molecular Sensing Data Analysis

This project implements machine learning approaches for analyzing molecular sensing data and predicting analyte concentrations. The system processes multi-dimensional time-series data from molecular sensors and applies various regression models to predict concentration levels.

## Overview

The project focuses on supervised learning for molecular sensing, using sensor signal data to predict concentrations of different molecular targets including:
- **Bacteria** (CFU concentrations)
- **IgG** (Immunoglobulin G with Au40 and HRPAB setups)
- **HRP** (Horseradish Peroxidase) which are **Control samples**

## Project Structure

```
├── No.1_preprocessing_xls.py    # Data preprocessing from Excel files
├── No.2_direct_regression_split_igg.py  # ML model training and evaluation
├── analyzed/                    # Processed data (JSON format)
├── Bacteria HRPAB/            # Raw bacteria sensing data
├── IgG Au40/                  # Raw IgG sensing data (Au40 setup)
├── IgG HRPAB/                 # Raw IgG sensing data (HRPAB setup)
├── Control/                   # Control experiment data HRP
├── results/                   # Model outputs and metrics
└── final_ver_report/          # Analysis reports and figures
```

## Key Features

### Data Processing
- Converts Excel sensor data to standardized JSON format
- Handles 200-cycle measurements with 61 time points each
- Applies concentration transformations: `log₁₀(concentration + 10⁻¹⁹)`
- Supports multiple molecular targets and experimental setups

### Machine Learning Models
- **Neural Networks**: Multi-layer perceptron (MLP) with configurable architecture
- **Tree-based**: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Other**: Support Vector Regression (SVR), K-Nearest Neighbors (KNR)

### Analysis Features
- Category-based holdout validation
- Cross-split robustness testing with multiple random seeds
- Comprehensive metrics calculation (R², MSE, RMSE, MAE, Pearson correlation)
- LDL (Limit of Detection) analysis
- Extensive visualization suite

## Quick Start

### 1. Data Preprocessing
```bash
python No.1_preprocessing_xls.py
```
Processes raw Excel files and saves to `analyzed/preprocessed/` as JSON.

### 2. Run ML Analysis
```bash
# Single algorithm (MLP)
python No.2_direct_regression_split_igg.py --algorithms mlp

# Multiple algorithms
python No.2_direct_regression_split_igg.py --algorithms mlp,rf,xgboost

# Custom configuration
python No.2_direct_regression_split_igg.py \
    --algorithms mlp \
    --hidden_dims 512,256,128 \
    --dropout_rate 0.18 \
    --learning_rate 0.003 \
    --trial_seeds 8,80,239,294,310
```

### 3. Results
Results are saved in `results/direct_regression_category_holdout_{algorithm}_{suffix}/`:
- Model predictions and metrics (CSV/JSON)
- Training history plots
- Scatter plots and error analysis
- Cross-split summaries

## Model Configuration

The MLP model uses optimized hyperparameters:
- **Hidden layers**: [512, 256, 128]
- **Dropout rate**: 0.18
- **Learning rate**: 0.003
- **Batch size**: 16
- **Robust trial seeds**: [8, 80, 239, 294, 310]

## Key Metrics

- **R² Score**: Coefficient of determination
- **RMSE**: Root mean squared error
- **Pearson Correlation**: Linear correlation coefficient
- **LDL Analysis**: Limit of detection calculations
- **Category-specific**: Separate metrics for each molecular target

## Dependencies

Core requirements:
- Python 3.7+
- PyTorch
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- openpyxl, xlrd

Optional (for advanced algorithms):
- xgboost
- catboost
- lightgbm

## Citation

If you use this code, please cite the related research on weak supervised representation learning for molecular sensing applications.

(Raw data supporting this research is available through controlled access due to proprietary sensor technology and ongoing patent considerations. For collaboration opportunities and data access, please contact xiaoaoshi@uchicago.edu and ruiding@uchicago.edu with your institutional affiliation and research objectives)
