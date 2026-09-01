# Predictive Hard Drive Failure Modeling for SRE Operations
### A Machine Learning Approach to Infrastructure Reliability

[![HarvardX Professional Certificate in Data Science](https://img.shields.io/badge/HarvardX-PH125.9x-A51C30.svg)](https://harvardonline.harvard.edu/program/data-science)
[![R-Project](https://img.shields.io/badge/Language-R_4.x-276DC3.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)
[![OS](https://img.shields.io/badge/OS-Zorin%20OS-blueviolet.svg)](https://zorin.com/os/)

---

## Executive Summary

Modern cloud storage infrastructure relies on high-density hard disk drive (HDD) deployments to manage exabyte-scale data. In a **Site Reliability Engineering (SRE)** context, unannounced drive failures induce operational toil, saturate network bandwidth during ungraceful RAID rebuilds, and risk Service Level Objective (SLO) violations.

This repository presents an end-to-end, cost-sensitive machine learning framework using Backblaze Q1 2024 quarterly telemetry (**>13 million observations**) to forecast drive failures prior to catastrophic breakdown. By capturing non-linear drive degradation dynamics over an optimized **14-day failure horizon ($H = 14$)**, the proposed system enables automated orchestrators to gracefully evacuate data (*drain mode*) while preserving error budgets.

## Key Deliverables & Project Architecture

This repository is organized to distinguish between the development pipeline, model execution, and the final academic presentation:

* [Predictive_HardDrive_Failure_modeling_for_SRE_Operations.pdf](./Predictive_HardDrive_Failure_modeling_for_SRE_Operations.pdf) : **Final Publication.** The formal academic and technical report detailing the SRE failure prediction methodology, cost-matrix optimization, and empirical findings in a publication-ready layout.
* [Predictive_HardDrive_Failure_modeling_for_SRE_Operations.Rmd](./Predictive_HardDrive_Failure_modeling_for_SRE_Operations.Rmd) : **The Refined Version.** The primary R Markdown source document used to render the final manuscript, seamlessly integrating code, statistical outputs, and analytical discussions.
* [R/main.R](./R/main.R) : **The Laboratory.** The main orchestrator script executing the full end-to-end "under the hood" workflow — from automated telemetry ingestion and temporal feature engineering ($\Delta_7$) to XGBoost training and SRE cost-sensitive threshold tuning.

## Key Highlights & Empirical Results

* **Optimized Horizon ($H=14$):** Engineered temporal rolling metrics ($\Delta_7$) across critical S.M.A.R.T. indicators (`smart_5`, `smart_187`, `smart_197`) to capture failure trajectories.
* **Extreme Imbalance Management:** Trained on extreme class imbalance ($\approx 0.04\%$ daily failure prevalence), achieving a **129x precision lift** over random baseline.
* **Cost-Sensitive Decision Thresholds:** Applied an enterprise SRE cost matrix ($C_{FN} = \$500$ vs $C_{FP} = \$10$) to optimize emergency on-call toil, network rebuild overhead, and hardware failure costs.
* **Resource-Constrained Pipeline:** Execution optimized under strict "Design by Constraint" resource limits (6 GB RAM, 4 vCPUs) using `data.table` and lightweight memory management.

### Model Performance Comparison

| Model Architecture | AUC-ROC | PR-AUC | Optimal Threshold ($\tau^*$) | Total Operational Cost |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Regression)** | 0.7210 | 0.0082 | 0.05 | \$142,500 |
| **XGBoost Classifier (Proposed)** | **0.8364** | **0.0516** | **0.012** | **\$38,210** |

## Visualizing Pipeline Performance & SRE Impact

| Performance Metrics (ROC & PR) | Cost-Sensitive Threshold Optimization |
| :---: | :---: |
| ![Model Performance](./outputs/figures/fig7_roc_curves_comparison.png) | ![SRE Cost Matrix](./outputs/figures/fig9_sre_cost_optimization.png) |
| **Model Discrimination:** XGBoost achieves **AUC-ROC 0.8364** and **PR-AUC 0.0516** (~129x precision lift) under extreme class imbalance ($0.04\%$). | **SRE Economic Optimization:** Threshold tuning minimizes total operational costs ($C_{FN} = \$500$ vs $C_{FP} = \$10$) by balancing emergency toil and proactive data evacuation. |

## Repository Structure

```text
.
├── R/                                                                            # Modular R pipeline scripts
│   ├── 00_download_and_ingest.R                                                  # Telemetry download, extraction & memory-optimized ingestion
│   ├── 01_eda_and_processing.R                                                   # Feature engineering (Delta_7) & data preprocessing
│   ├── 01b_exploratory_data_analysis.R                                           # EDA, failure distributions & S.M.A.R.T. trends analysis
│   ├── 02_model_training.R                                                       # Model training (Logistic Regression & XGBoost Classifier)
│   ├── 03_model_evaluation.R                                                     # Performance evaluation (ROC & Precision-Recall curves)
│   ├── 04_sre_cost_analysis.R                                                    # SRE cost matrix optimization ($C_FN vs $C_FP) & threshold tuning
│   ├── 05_model_interpretability.R                                               # Feature importance & SHAP model explainability analysis
│   └── main.R                                                                    # Master orchestrator script (executes full end-to-end pipeline)
├── outputs/                                                                      # Generated pipeline artifacts
│   ├── figures/                                                                  # High-resolution PNG plots for report and README
│   │   ├── fig1_class_imbalance.png
│   │   ├── fig2_failure_rate_by_model.png
│   │   ├── fig3_smart_features_distribution.png
│   │   ├── fig4_correlation_heatmap.png
│   │   ├── fig5_temporal_failure_trend.png
│   │   ├── fig6_failure_rate_by_capacity.png
│   │   ├── fig7_roc_curves_comparison.png
│   │   ├── fig8_precision_recall_curve.png
│   │   ├── fig9_sre_cost_optimization.png
│   │   └── fig10_shap_feature_importance.png
│   └── models/                                                                   # Saved model artifacts and serialized binaries
│       └── model_xgboost.model
├── Predictive_HardDrive_Failure_modeling_for_SRE_Operations.Rmd                  # Comprehensive R Markdown manuscript source
├── Predictive_HardDrive_Failure_modeling_for_SRE_Operations.pdf                  # Rendered PDF report (Final HarvardX Capstone deliverable)
├── capstone_report_files/                                                        # Auxiliary LaTeX rendering assets for Rmd compilation
├── backblaze-sre-failure-prediction.Rproj                                        # RStudio project configuration file
├── LICENSE                                                                       # MIT License
└── README.md                                                                     # Project overview & documentation
```

## Quick Start & Pipeline Execution
### Prerequisites

Ensure you have R 4.x installed along with the following required libraries:
```
install.packages(c("tidyverse", "data.table", "xgboost", "pROC", "PRROC", "knitr", "rmarkdown"))
```
### Reproducing the Pipeline

You can run the full pipeline sequentially or execute the orchestrator script:
```
# Clone the repository
git clone [https://github.com/efhache/backblaze-sre-failure-prediction.git](https://github.com/efhache/backblaze-sre-failure-prediction.git)
cd backblaze-sre-failure-prediction

# Execute the master script to run pipeline & build report
Rscript main.R
```

## Dataset Description

The models are trained and evaluated on open-source enterprise telemetry provided by Backblaze:

* **Source:** Backblaze Hard Drive Data and Stats
* **Scope:** Multi-year daily drive operational status, failure indicators, and S.M.A.R.T. telemetry attributes.
* **Preprocessing:** Filtering for enterprise models, temporal partitioning (out-of-time validation split), and missing value imputations.
* **Validation Strategy:** Out-of-Time (OOT) temporal validation split (Training: Jan–Feb 2024 | Testing: March 2024) to eliminate data leakage.
