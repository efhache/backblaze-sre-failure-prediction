# Predictive Hard Drive Failure Modeling for SRE Operations
### A Machine Learning Approach to Infrastructure Reliability

[![HarvardX Professional Certificate in Data Science](https://img.shields.io/badge/HarvardX-PH125.9x-A51C30.svg)](https://harvardonline.harvard.edu/program/data-science)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![R-Project](https://img.shields.io/badge/Language-R%204.x-blue.svg)](https://www.r-project.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)
[![OS](https://img.shields.io/badge/OS-Zorin%20OS-blueviolet.svg)](https://zorin.com/os/)
---

## Executive Summary

Modern cloud storage infrastructure relies on high-density hard disk drive (HDD) deployments to manage exabyte-scale data. In a **Site Reliability Engineering (SRE)** context, unannounced drive failures induce operational toil, saturate network bandwidth during ungraceful RAID rebuilds, and risk Service Level Objective (SLO) violations.

This repository presents an end-to-end, cost-sensitive machine learning framework using multi-year **Backblaze daily telemetry** (>13 million observations) to forecast drive failures prior to catastrophic breakdown. By capturing non-linear drive degradation dynamics over an optimized **7-day failure horizon ($H = 7$)**, the proposed system enables automated orchestrators to gracefully evacuate data (*drain mode*) while preserving error budgets.

---

## Key Highlights & Empirical Results

* **Optimized Horizon ($H=7$):** Engineered temporal rolling metrics ($\Delta_7$) across critical S.M.A.R.T. indicators (`smart_5`, `smart_187`, `smart_197`) to capture failure trajectories.
* **Extreme Imbalance Management:** Trained on extreme class imbalance ($\approx 0.04\%$ daily failure prevalence), achieving an **AUC-ROC of 0.8364** and a **PR-AUC of 0.0516** (~129x precision lift over baseline).
* **Cost-Sensitive Decision Thresholds:** Applied an enterprise SRE cost matrix ($C_{FN} = \$500$ vs $C_{FP} = \$10$) to optimize emergency on-call toil, network rebuild overhead, and hardware failure costs.
* **Resource-Constrained Pipeline:** Execution optimized under strict "Design by Constraint" resource limits (6 GB RAM, 4 vCPUs) using data.table and lightweight memory management.

---

## Visualizing Pipeline Performance & SRE Impact

| Performance Metrics (ROC & PR) | Cost-Sensitive Threshold Optimization |
| :---: | :---: |
| ![Model Performance](./outputs/figures/fig7_roc_curves_comparison.png) | ![SRE Cost Matrix](./outputs/figures/fig9_sre_cost_optimization.png) |
| **Model Discrimination:** XGBoost achieves **AUC-ROC 0.8364** and **PR-AUC 0.0516** (~129x precision lift) under extreme class imbalance ($0.04\%$). | **SRE Economic Optimization:** Threshold tuning minimizes total operational costs ($C_{FN} = \$500$ vs $C_{FP} = \$10$) by balancing emergency toil and proactive data evacuation. |

---

## Repository Structure

```text
.
├── scripts/
│   ├── 00_ingestion.R          # Telemetry extraction, filtering & memory management
│   ├── 01_eda.R                # Exploratory Data Analysis & failure distribution
│   ├── 02_feature_engineering.R# Temporal lag metrics (Delta_7) & target labeling
│   ├── 03_modeling.R           # Model training (Logistic Regression, XGBoost)
│   └── 04_evaluation.R         # ROC/PR curves, cost-matrix & SRE decision threshold
├── figures/                    # High-resolution generated plots for the manuscript
├── data/                       # Dataset directory (git-ignored)
├── main.R                      # Master pipeline orchestrator script
├── backblaze-sre-failure-prediction.Rmd # Comprehensive R Markdown source
├── backblaze-sre-failure-prediction.pdf # Final rendered PDF report
├── references.bib              # Bibliography and textbook citations
└── README.md                   # Repository overview
```
---
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
---
## Dataset Description

The models are trained and evaluated on open-source enterprise telemetry provided by Backblaze:

* **Source:** Backblaze Hard Drive Data and Stats
* **Scope:** Multi-year daily drive operational status, failure indicators, and S.M.A.R.T. telemetry attributes.
* **Preprocessing:** Filtering for enterprise models, temporal partitioning (out-of-time validation split), and missing value imputations.
