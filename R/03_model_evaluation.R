# ==============================================================================
# Script: 03_model_evaluation.R
# Project: Predictive Hard Drive Failure Modelling for SRE Operations
# Description: Model Evaluation (AUC-ROC, Precision-Recall, SRE Cost Curves)
# ==============================================================================

# 0. Load Libraries
required_packages <- c("data.table", "tidyverse", "pROC", "PRROC", "scales", "gridExtra")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

theme_set(theme_minimal(base_size = 12))

PATH_PROCESSED <- "data/processed/"
PATH_FIGS      <- "outputs/figures/"
PATH_METRICS   <- "outputs/metrics/"

if (!dir.exists(PATH_METRICS)) dir.create(PATH_METRICS, recursive = TRUE)

INPUT_PREDS <- file.path(PATH_PROCESSED, "test_predictions_q1_2024.rds")

cat("Loading test set predictions...\n")
preds_dt <- readRDS(INPUT_PREDS)

# ==============================================================================
# 1. Compute ROC Curves & AUC Metric
# ==============================================================================
cat("\n--- Calculating ROC & PR Metrics ---\n")

roc_lr  <- roc(preds_dt$target_14d, preds_dt$pred_lr, quiet = TRUE)
roc_xgb <- roc(preds_dt$target_14d, preds_dt$pred_xgb, quiet = TRUE)

auc_lr_val  <- auc(roc_lr)
auc_xgb_val <- auc(roc_xgb)

cat(sprintf("Baseline (Logistic Regression) AUC-ROC: %.4f\n", auc_lr_val))
cat(sprintf("Advanced (XGBoost)             AUC-ROC: %.4f\n", auc_xgb_val))

# ==============================================================================
# 2. Plotting ROC Curves Comparison
# ==============================================================================
df_roc_lr  <- data.frame(FPR = 1 - roc_lr$specificities, TPR = roc_lr$sensitivities, Model = sprintf("Logistic Regression (AUC = %.3f)", auc_lr_val))
df_roc_xgb <- data.frame(FPR = 1 - roc_xgb$specificities, TPR = roc_xgb$sensitivities, Model = sprintf("XGBoost (AUC = %.3f)", auc_xgb_val))

df_roc <- rbind(df_roc_lr, df_roc_xgb)

p_roc <- ggplot(df_roc, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  scale_color_manual(values = c("#e74c3c", "#2980b9")) +
  labs(
    title = "ROC Curves Comparison on Temporal Test Set (March 2024)",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity / Recall)",
    color = "Model"
  ) +
  theme(legend.position = "bottom")

ggsave(file.path(PATH_FIGS, "fig7_roc_curves_comparison.png"), plot = p_roc, width = 8, height = 6, dpi = 300)

# ==============================================================================
# 3. Precision-Recall Curve (Critical for Extreme Class Imbalance)
# ==============================================================================
pr_xgb <- pr.curve(
  scores.class0 = preds_dt[target_14d == 1, pred_xgb],
  scores.class1 = preds_dt[target_14d == 0, pred_xgb],
  curve = TRUE
)

df_pr <- data.frame(Recall = pr_xgb$curve[, 1], Precision = pr_xgb$curve[, 2])

p_pr <- ggplot(df_pr, aes(x = Recall, y = Precision)) +
  geom_line(color = "#27ae60", size = 1.2) +
  labs(
    title = sprintf("XGBoost Precision-Recall Curve (PR-AUC = %.4f)", pr_xgb$auc.integral),
    subtitle = "Essential evaluation metric given the 0.04% positive class imbalance",
    x = "Recall (Sensitivity)",
    y = "Precision"
  )

ggsave(file.path(PATH_FIGS, "fig8_precision_recall_curve.png"), plot = p_pr, width = 8, height = 6, dpi = 300)

# ==============================================================================
# 4. Export Summary Table for Thesis Report
# ==============================================================================
summary_table <- data.table(
  Model = c("Logistic Regression (Baseline)", "XGBoost Classifier"),
  AUC_ROC = c(auc_lr_val, auc_xgb_val),
  PR_AUC  = c(NA, pr_xgb$auc.integral),
  Test_Observations = nrow(preds_dt),
  Test_Failures = sum(preds_dt$target_14d)
)

write.csv(summary_table, file.path(PATH_METRICS, "model_performance_summary.csv"), row.names = FALSE)

rm(roc_lr, roc_xgb, pr_xgb, df_roc_lr, df_roc_xgb, df_roc, df_pr, p_roc, p_pr)
gc(verbose = FALSE)
cat("\nSummary of results exported to 'outputs/metrics/model_performance_summary.csv'\n")

cat("Phase 4 Model Evaluation completed successfully!\n")