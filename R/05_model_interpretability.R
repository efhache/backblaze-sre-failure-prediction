# ==============================================================================
# Script: 05_model_interpretability.R
# Project: Predictive Hard Drive Failure Modelling for SRE Operations
# Description: Model Interpretability (Native XGBoost SHAP & Feature Importance)
# Standard: HarvardX / edX Data Science Capstone Standards
# ==============================================================================

# 0. Load Libraries
required_packages <- c("data.table", "tidyverse", "xgboost", "gridExtra", "scales")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

theme_set(theme_minimal(base_size = 12))

DATASET_TAG    <- "Q1_2024"
PATH_PROCESSED <- "data/processed/"
PATH_MODELS    <- "outputs/models/"
PATH_FIGS      <- "outputs/figures/"
PATH_METRICS   <- "outputs/metrics/"

if (!dir.exists(PATH_FIGS)) dir.create(PATH_FIGS, recursive = TRUE)
if (!dir.exists(PATH_METRICS)) dir.create(PATH_METRICS, recursive = TRUE)

# ==============================================================================
# 1. Load Data and Trained XGBoost Model
# ==============================================================================
cat("Loading trained XGBoost model...\n")
model_path <- file.path(PATH_MODELS, "model_xgboost.model")
model_xgb  <- xgb.load(model_path)

cat("Loading and preparing test set dataset...\n")
input_filepath <- file.path(PATH_PROCESSED, sprintf("dt_processed_%s.rds", tolower(DATASET_TAG)))
dt <- readRDS(input_filepath)
dt[, date := as.IDate(date)]

# Split temporel identique au Script 02
cutoff_date <- as.IDate("2024-03-01")
test_dt     <- dt[date >= cutoff_date]
rm(dt)
gc()

# Calcul des variables dérivées identiques
test_dt[, capacity_tb := capacity_bytes / 1e12]

feature_cols <- c(
  "capacity_tb", "smart_5_raw", "smart_9_raw", "smart_187_raw", 
  "smart_188_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw",
  "smart_5_delta7", "smart_187_delta7", "smart_197_delta7"
)

# Imputation des NA par 0
for (j in feature_cols) {
  set(test_dt, which(is.na(test_dt[[j]])), j, 0)
}

X_test <- as.matrix(test_dt[, ..feature_cols])

# ==============================================================================
# 2. Native XGBoost Feature Importance (Gain & Cover)
# ==============================================================================
cat("Calculating native XGBoost feature importance...\n")
importance_matrix <- xgb.importance(feature_names = feature_cols, model = model_xgb)

write.csv(importance_matrix, file.path(PATH_METRICS, "xgboost_feature_importance.csv"), row.names = FALSE)

# ==============================================================================
# 3. Native Fast-SHAP Calculation via XGBoost Engine
# ==============================================================================
cat("Calculating SHAP values using native XGBoost C++ engine...\n")

set.seed(42)
sample_size <- min(20000, nrow(X_test))
sample_idx  <- sample(seq_len(nrow(X_test)), size = sample_size)
X_sample    <- X_test[sample_idx, ]

# Contribution SHAP native par arbre XGBoost
shap_contrib <- predict(model_xgb, newdata = X_sample, predcontrib = TRUE)

# Extraction des contributions (sans la colonne BIAS)
shap_matrix <- shap_contrib[, -ncol(shap_contrib)]

# Mean Absolute SHAP Value par variable
mean_shap <- colMeans(abs(shap_matrix))
shap_summary_df <- data.table(
  Feature = names(mean_shap),
  Mean_Abs_SHAP = mean_shap
)[order(-Mean_Abs_SHAP)]

write.csv(shap_summary_df, file.path(PATH_METRICS, "shap_feature_importance.csv"), row.names = FALSE)

# ==============================================================================
# 4. Generate Figure 10: Dual Interpretability Plot
# ==============================================================================
cat("Generating Figure 10 (Gain Importance & SHAP Mean Contribution)...\n")

# Plot A: Importance par Gain (%)
top_gain_df <- head(importance_matrix, 10)
p1 <- ggplot(top_gain_df, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#2c3e50", width = 0.7) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "A: Global Feature Importance (Gain)",
    subtitle = "Relative contribution of SMART features to overall decision tree splits",
    x = "SMART Feature",
    y = "Gain (%)"
  )

# Plot B: Mean Absolute SHAP Value
top_shap_df <- head(shap_summary_df, 10)
p2 <- ggplot(top_shap_df, aes(x = reorder(Feature, Mean_Abs_SHAP), y = Mean_Abs_SHAP)) +
  geom_col(fill = "#27ae60", width = 0.7) +
  coord_flip() +
  labs(
    title = "B: Global SHAP Feature Importance (|Mean SHAP Value|)",
    subtitle = "Average absolute impact of SMART attributes on model log-odds predictions",
    x = "SMART Feature",
    y = "|Mean SHAP Value|"
  )

p_combined <- grid.arrange(p1, p2, ncol = 1)

ggsave(
  file.path(PATH_FIGS, "fig10_shap_feature_importance.png"),
  plot = p_combined,
  width = 9,
  height = 10,
  dpi = 300
)

cat("\nFeature importance and SHAP analysis completed successfully!\n")
cat("Summary metrics exported to 'outputs/metrics/'\n")
cat("Figure saved to: outputs/figures/fig10_shap_feature_importance.png\n")