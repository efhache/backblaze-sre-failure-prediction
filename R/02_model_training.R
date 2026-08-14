# ==============================================================================
# Script: 02_model_training.R
# Project: Predictive Hard Drive Failure Modelling for SRE Operations
# Description: Strict temporal train/test split, model training (Baseline & XGBoost)
# ==============================================================================

# 0. Load Libraries
required_packages <- c("data.table", "tidyverse", "lubridate", "xgboost", "pROC")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

options(scipen = 999)
gc()

# ==============================================================================
# 1. Global Configuration & Paths
# ==============================================================================
DATASET_TAG    <- "Q1_2024"
PATH_PROCESSED <- "data/processed/"
PATH_MODELS    <- "outputs/models/"

if (!dir.exists(PATH_MODELS)) dir.create(PATH_MODELS, recursive = TRUE)

INPUT_FILEPATH <- file.path(PATH_PROCESSED, sprintf("dt_processed_%s.rds", tolower(DATASET_TAG)))

# ==============================================================================
# 2. Loading Processed Dataset
# ==============================================================================
cat(sprintf("Loading processed dataset from %s...\n", INPUT_FILEPATH))
dt <- readRDS(INPUT_FILEPATH)

dt[, date := as.IDate(date)]

# ==============================================================================
# 3. Strict Temporal Train / Test Split (Prevention of Temporal Leakage)
# ==============================================================================
cutoff_date <- as.IDate("2024-03-01")

cat(sprintf("\n--- Performing Strict Temporal Split at %s ---\n", cutoff_date))

train_dt <- dt[date < cutoff_date]
test_dt  <- dt[date >= cutoff_date]

cat(sprintf("Train Set: %s rows | Failure Window (14d) Count: %s\n", 
            format(nrow(train_dt), big.mark = " "), sum(train_dt$target_14d)))
cat(sprintf("Test Set : %s rows | Failure Window (14d) Count: %s\n", 
            format(nrow(test_dt), big.mark = " "), sum(test_dt$target_14d)))

rm(dt)
gc()

# ==============================================================================
# 4. Feature Selection & Preprocessing
# ==============================================================================
cat("\n--- Preparing features & handling missing values ---\n")

train_dt[, capacity_tb := capacity_bytes / 1e12]
test_dt[,  capacity_tb := capacity_bytes / 1e12]

feature_cols <- c(
  "capacity_tb", "smart_5_raw", "smart_9_raw", "smart_187_raw", 
  "smart_188_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw",
  "smart_5_delta7", "smart_187_delta7", "smart_197_delta7"
)

target_col <- "target_14d"

# Impute NA values with 0 directly in-place
for (j in feature_cols) {
  set(train_dt, which(is.na(train_dt[[j]])), j, 0)
  set(test_dt,  which(is.na(test_dt[[j]])),  j, 0)
}

# ==============================================================================
# 5. Baseline Model: Logistic Regression (Anti-Collinearity & RAM-Optimised)
# ==============================================================================
cat("\n--- Training Model 1: Logistic Regression (Baseline) ---\n")

set.seed(42)
pos_indices <- train_dt[target_14d == 1, which = TRUE]
neg_indices <- train_dt[target_14d == 0, which = TRUE]

# Subsample 20x negative instances relative to positive count (~80k rows)
sampled_neg_indices <- sample(neg_indices, size = length(pos_indices) * 20)
lr_train_dt <- train_dt[c(pos_indices, sampled_neg_indices)]

# 5.1 Automatic Detection & Dropping of Constant or Collinear Features for GLM
glm_features <- c()
for (col in feature_cols) {
  vals <- log1p(pmax(0, lr_train_dt[[col]]))
  if (sd(vals) > 0.0001) {
    glm_features <- c(glm_features, col)
  }
}

# Drop collinear columns with correlation > 0.98 in training slice
if (length(glm_features) > 1) {
  # Convert to matrix and clamp negative values without flattening matrix dimensions
  mat <- as.matrix(lr_train_dt[, ..glm_features])
  mat[mat < 0] <- 0
  
  cor_mat <- cor(log1p(mat))
  cor_mat[upper.tri(cor_mat, diag = TRUE)] <- 0
  
  collinear_cols <- glm_features[apply(abs(cor_mat) > 0.98, 2, any)]
  if (length(collinear_cols) > 0) {
    cat(sprintf("Dropping highly collinear features for GLM: %s\n", paste(collinear_cols, collapse = ", ")))
    glm_features <- setdiff(glm_features, collinear_cols)
  }
}

# 5.2 Scale Subsampled Train Set for GLM
scaled_train_dt <- copy(lr_train_dt[, ..glm_features])
scale_means <- list()
scale_sds   <- list()

for (col in glm_features) {
  raw_vals <- log1p(pmax(0, scaled_train_dt[[col]]))
  m_val <- mean(raw_vals)
  s_val <- sd(raw_vals)
  if (s_val == 0) s_val <- 1
  
  scale_means[[col]] <- m_val
  scale_sds[[col]]   <- s_val
  
  scaled_train_dt[, (col) := (raw_vals - m_val) / s_val]
}
scaled_train_dt[, (target_col) := lr_train_dt[[target_col]]]

# 5.3 Fit Logistic Regression
lr_formula <- as.formula(paste(target_col, "~", paste(glm_features, collapse = " + ")))

lr_model <- glm(
  formula = lr_formula, 
  data    = scaled_train_dt, 
  family  = binomial(link = "logit")
)

cat("Logistic Regression fitted successfully. Predicting on Test Set...\n")

# 5.4 Scale Test Set on-the-fly and Predict
scaled_test_dt <- data.table(matrix(0, nrow = nrow(test_dt), ncol = length(glm_features)))
colnames(scaled_test_dt) <- glm_features

for (col in glm_features) {
  t_vals <- log1p(pmax(0, test_dt[[col]]))
  scaled_test_dt[, (col) := (t_vals - scale_means[[col]]) / scale_sds[[col]]]
}

test_dt[, pred_lr := predict(lr_model, newdata = scaled_test_dt, type = "response")]
cat("Baseline predictions generated successfully.\n")

# Free memory from intermediate objects
rm(lr_train_dt, scaled_train_dt, scaled_test_dt)
gc()

# ==============================================================================
# 6. Advanced Model: XGBoost Classifier (Full Train Set)
# ==============================================================================
cat("\n--- Training Model 2: XGBoost Classifier (Full 8.5M Train Set) ---\n")

# Prepare DMatrix objects
dtrain <- xgb.DMatrix(
  data  = as.matrix(train_dt[, ..feature_cols]), 
  label = train_dt[[target_col]]
)

dtest <- xgb.DMatrix(
  data  = as.matrix(test_dt[, ..feature_cols]), 
  label = test_dt[[target_col]]
)

# Calculate class imbalance ratio for scale_pos_weight
neg_count <- sum(train_dt[[target_col]] == 0)
pos_count <- sum(train_dt[[target_col]] == 1)
scale_pos_weight_val <- neg_count / pos_count

xgb_params <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = "auc",
  max_depth        = 6,
  eta              = 0.1,
  scale_pos_weight = scale_pos_weight_val,
  nthread          = 2
)

cat(sprintf("Class Imbalance Ratio (scale_pos_weight): %.2f\n", scale_pos_weight_val))

set.seed(42)
xgb_model <- xgb.train(
  params                = xgb_params,
  data                  = dtrain,
  nrounds               = 100,
  watchlist             = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,
  print_every_n         = 20
)

# Predict probabilities on FULL Test Set
test_dt[, pred_xgb := predict(xgb_model, dtest)]

rm(dtrain, dtest)
gc()

# ==============================================================================
# 7. Save Models and Test Predictions
# ==============================================================================
cat("\n--- Saving Trained Models & Test Predictions ---\n")

saveRDS(lr_model, file = file.path(PATH_MODELS, "model_logistic_regression.rds"))
xgb.save(xgb_model, file.path(PATH_MODELS, "model_xgboost.model"))

output_preds_path <- file.path(PATH_PROCESSED, sprintf("test_predictions_%s.rds", tolower(DATASET_TAG)))
saveRDS(test_dt[, .(serial_number, date, model, target_14d, pred_lr, pred_xgb)], file = output_preds_path)

cat(sprintf("Predictions successfully exported to %s\n", output_preds_path))
cat("Phase 3 Training pipeline completed successfully!\n")
gc()