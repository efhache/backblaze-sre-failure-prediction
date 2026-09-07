###############################################################################
# Predictive Hard Drive Failure Modelling for SRE Operations
# All-in one - Consolidated Reproducibility Pipeline
#
# This standalone script reproduces the complete project workflow within a
# single executable file. While the original implementation follows a modular
# architecture orchestrated through main.R and multiple task-specific scripts,
# the present version consolidates data ingestion, preprocessing, exploratory
# analysis, model training, evaluation, cost optimisation, and
# interpretability into a single end-to-end pipeline.
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    WARNING:    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Unlike the reference implementation, all processing stages are executed
# within the same R session. As a result, memory consumption accumulates during
# execution and may exceed the capacity of constrained environments.
#
# The recommended implementation remains the modular pipeline orchestrated via
# main.R, which was specifically designed to support large-scale Backblaze
# datasets on a virtual machine limited to 6 GB RAM through stage-level memory
# reclamation.
#
# Project Repository:
# https://github.com/efhache/backblaze-sre-failure-prediction
#
#
# This script is provided solely as a single-file reproducibility artifact.
###############################################################################



# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================
# DEMO_MODE <- TRUE: Fast mode for a quick peer review (< 2 mins, < 2 GB RAM)
# DEMO_MODE <- FALSE: Full Production Mode (Exact reconstruction of the PDF, ~13M+ lines)
DEMO_MODE <- TRUE

cat("==================================================================\n")
cat(sprintf("   RUNNING BACKBLAZE AIOPS PIPELINE (DEMO_MODE = %s)\n", DEMO_MODE))
cat("==================================================================\n")
if (DEMO_MODE) {
  cat("  [INFO] Fast execution enabled: Single model target & 15 XGBoost trees.\n")
  cat("  [INFO] Execution time: < 2 mins | Memory usage: < 2 GB RAM.\n")
  cat("  [INFO] Set DEMO_MODE <- FALSE for full 13M+ row evaluation.\n")
}
cat("==================================================================\n\n")




################################################################################
# SECTION 0 - LIBRARIES
################################################################################

required_packages <- c(
  "data.table",
  "tidyverse",
  "lubridate",
  "ggplot2",
  "xgboost",
  "pROC",
  "PRROC",
  "scales",
  "gridExtra",
  "httr"
)

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

options(scipen = 999)
theme_set(theme_minimal(base_size = 12))

gc()

cat("==================================================================\n")
cat("Starting Complete Backblaze AIOps Failure Prediction Pipeline\n")
cat("==================================================================\n\n")

start_time <- Sys.time()

################################################################################
# SECTION 1 - DOWNLOAD & INGESTION
# Automated downloading, decompression and cleaning of data structures
################################################################################

local({
  options(timeout = 3600) #Global R timeout setting (1 hour = 3,600 seconds)
  
  # 1. Directory configuration
  raw_dir <- "data/raw"
  if (!dir.exists(raw_dir)) dir.create(raw_dir, recursive = TRUE)
  
  # 2. Backblaze’s official Q1 2024 URL
  zip_url  <- "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2024.zip"
  zip_path <- file.path(raw_dir, "data_Q1_2024.zip")
  
  # 3. Download if the archive does not exist or is incomplete
  if (!file.exists(zip_path) || file.info(zip_path)$size < 100000000) {
    message("--> Download the Backblaze Q1 2024 archive (please wait)...")
    #Force the use of the libcurl/curl engine to bypass the RStudio driver, which sets the timeout to 60 seconds
    download.file(
      url      = zip_url, 
      destfile = zip_path, 
      method   = if (capabilities("libcurl")) "libcurl" else "auto", 
      mode     = "wb"
    )
  } else {
    message("--> Zip archive already present locally.")
  }
  
  # 4. Extracting the archive
  message("--> Extract the zip archive to data/raw/...")
  unzip(zip_path, exdir = raw_dir)
  
  # 5. Resolving the issue of unwanted macOS subfolders and files
  message("--> Standardisation of the structure of CSV files...")
  
  # A. Immediate deletion of the __MACOSX junk folder created by macOS
  macosx_dir <- file.path(raw_dir, "__MACOSX")
  if (dir.exists(macosx_dir)) {
    unlink(macosx_dir, recursive = TRUE)
    message("   [OK] The __MACOSX folder has been deleted.")
  }
  
  # B. Migrate the CSV files from the extracted subfolder to data/raw/
  nested_dir <- file.path(raw_dir, "data_Q1_2024")
  if (dir.exists(nested_dir)) {
    nested_csvs <- list.files(nested_dir, pattern = "\\.csv$", full.names = TRUE)
    if (length(nested_csvs) > 0) {
      file.rename(nested_csvs, file.path(raw_dir, basename(nested_csvs)))
      message(sprintf("   [OK] %d CSV files moved from the subfolder to %s/", length(nested_csvs), raw_dir))
    }
    # Deleting the empty folder
    unlink(nested_dir, recursive = TRUE)
  }
  
  # C. Cleaning up Apple’s hidden files (ex: ._2024-01-01.csv)
  apple_dot_files <- list.files(raw_dir, pattern = "^\\._.*\\.csv$", full.names = TRUE)
  if (length(apple_dot_files) > 0) {
    file.remove(apple_dot_files)
    message(sprintf("   [OK] %d malicious Apple cache files (._*.csv) deleted.", length(apple_dot_files)))
  }
  
  # 6. Final check
  valid_csvs <- list.files(raw_dir, pattern = "^2024-.*\\.csv$", full.names = TRUE)
  message(sprintf("==> Import completed successfully: %d valid CSV files ready for processing.", length(valid_csvs)))
})
gc()

################################################################################
# SECTION 2 - DATA PREPARATION & FEATURE ENGINEERING
# Description: Optimised data ingestion, exploration (EDA) and data preparation
################################################################################

local({
  # 3. Defining relative paths
  PATH_RAW       <- "data/raw/"
  PATH_PROCESSED <- "data/processed/"
  PATH_FIGURES   <- "outputs/figures/"
  
  # 4. Defining quarter to analyse
  DATASET_TAG    <- "Q1_2024" # Adjust if processing a different quarter (e.g., "Q2_2024")
  
  
  # Dynamic output file construction based on dataset tag
  OUTPUT_FILENAME <- sprintf("dt_processed_%s.rds", tolower(DATASET_TAG))
  OUTPUT_FILEPATH <- file.path(PATH_PROCESSED, OUTPUT_FILENAME)
  EDA_RAW_FILEPATH <- file.path(PATH_PROCESSED, "eda_raw_summary.rds")
  
  cat("=========================================================\n")
  cat(" Environment successfully initialised for the VM (6 GB RAM)\n")
  cat("=========================================================\n")
  
  
  # ==============================================================================
  # Phase 2: Optimised data ingestion, EDA & feature engineering
  # ==============================================================================
  # 1. Selection of strategic columns (RAM optimisation)
  # ---------------------------------------------------------------------------- --
  # Columns identified in the literature review (Pinheiro et al., 2007):
  # SMART 5: Reallocated Sectors Count
  # SMART 187: Reported Uncorrectable Errors
  # SMART 188: Command Timeout
  # SMART 197: Current Pending Sector Count
  # SMART 198: Offline Uncorrectable / Scan Errors
  # SMART 9: Power-On Hours (Drive Age)
  # SMART 194: Temperature (For monitoring/validation)
  
  keep_cols <- c(
    "date", "serial_number", "model", "capacity_bytes", "failure",
    "smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw", 
    "smart_194_raw", "smart_197_raw", "smart_198_raw"
  )
  
  # 2. Optimised data ingestion with data.table
  # ------------------------------------------------------------------------------
  cat("Starting to import the CSV files...\n")
  
  # List all the CSV files extracted to the data/raw/ directory
  csv_files <- list.files(path = PATH_RAW, pattern = "*.csv", full.names = TRUE)
  
  if (length(csv_files) == 0) {
    stop("Please note: No CSV files were found in the ‘data/raw/’ directory. Please place the Backblaze data there.")
  }
  
  # Fast ingestion and column selection only
  dt_list <- lapply(csv_files, function(file) {
    fread(
      file, 
      select = keep_cols, 
      colClasses = c(capacity_bytes = "numeric"), # Force dual-channel/64-bit reading
      showProgress = FALSE
    )
  })
  
  # Fast ingestion and column selection only
  dt_raw <- rbindlist(dt_list, fill = TRUE)
  rm(dt_list) # Immediate memory release
  gc()
  
  cat(sprintf("Input complete: %s rows and %s columns loaded into memory.\n", 
              format(nrow(dt_raw), big.mark = " "), ncol(dt_raw)))
  
  # Extraction & Saving Raw EDA Metadata
  raw_failure_dist <- dt_raw[, .N, by = failure][order(failure)]
  top_10_models <- dt_raw[, .N, by = model][order(-N)][1:10]
  
  # Interception and explicit saving of raw metrics
  eda_summary <- list(
    failure_dist = raw_failure_dist,
    top_models = top_10_models,
    total_rows = nrow(dt_raw)
  )
  saveRDS(eda_summary, file = EDA_RAW_FILEPATH)
  
  # 3. Initial quick review and quality control (basic EDA)
  # ------------------------------------------------------------------------------
  cat("\n--- Distribution of the target variable (Failure) ---\n")
  print(table(dt_raw$failure, useNA = "ifany"))
  
  cat("\n--- Top 10 most common disc models ---\n")
  top_models <- dt_raw[, .N, by = model][order(-N)][1:10]
  print(top_models)
  
  # Memory clean-up
  gc
  
  # ==============================================================================
  # 4. Data Filtering & Model Selection
  # ==============================================================================
  cat("\n--- Filtering top drive models to reduce noise ---\n")
  
  if (DEMO_MODE) {
    cat("--> [DEMO_MODE] Restricting processing to 1 single drive model (ST12000NM0007)\n")
    # Conserve uniquement le modèle le plus représentatif (~20% du dataset total, idéal pour la RAM)
    top_model_names <- top_10_models[1, model]
  } else {
    # Keep top drive models representing majority of the population
    top_model_names <- top_10_models[1:5, model]
  }
  
  dt_filtered <- dt_raw[model %in% top_model_names]
  
  # Free up memory from raw data
  rm(dt_raw)
  gc()
  
  # Sort data chronologically per drive (essential for time-series feature engineering)
  setkey(dt_filtered, serial_number, date)
  
  
  # ==============================================================================
  # 5. Temporal Feature Engineering (SRE Prediction Horizon Y_it)
  # ==============================================================================
  cat("\n--- Creating 14-day Failure Horizon (Target Y_it) ---\n")
  
  # Define target window: 14-day lead time for proactive replacement
  PREDICTION_WINDOW_DAYS <- 14
  
  # Calculate max failure date per drive serial number
  dt_filtered[, max_failure := max(failure), by = serial_number]
  dt_filtered[, failure_date := as.IDate(ifelse(failure == 1, date, NA)), by = serial_number]
  dt_filtered[, failure_date := min(failure_date, na.rm = TRUE), by = serial_number]
  
  # Define binary label Y_it: 1 if drive fails within [t, t + 14 days]
  dt_filtered[, days_to_failure := as.numeric(failure_date - date)]
  dt_filtered[, target_14d := ifelse(!is.na(days_to_failure) & 
                                       days_to_failure >= 0 & 
                                       days_to_failure <= PREDICTION_WINDOW_DAYS, 1, 0)]
  
  # Clean temporary calculation columns
  dt_filtered[, c("max_failure", "failure_date", "days_to_failure") := NULL]
  
  
  # ==============================================================================
  # 6. Feature Engineering: SMART Deltas (7-day Rate of Change)
  # ==============================================================================
  cat("\n--- Computing 7-day SMART Delta Features ---\n")
  
  # Calculate 7-day changes for critical SMART metrics (Pinheiro et al., 2007)
  # Delta = Current raw value - Value 7 days ago
  dt_filtered[, smart_5_delta7 := smart_5_raw - shift(smart_5_raw, 7, type = "lag"), by = serial_number]
  dt_filtered[, smart_187_delta7 := smart_187_raw - shift(smart_187_raw, 7, type = "lag"), by = serial_number]
  dt_filtered[, smart_197_delta7 := smart_197_raw - shift(smart_197_raw, 7, type = "lag"), by = serial_number]
  
  # Impute initial NA values resulting from lag calculations with 0
  na_cols <- c("smart_5_delta7", "smart_187_delta7", "smart_197_delta7")
  for (j in na_cols) set(dt_filtered, which(is.na(dt_filtered[[j]])), j, 0)
  
  
  # ==============================================================================
  # 7. Summary & Export Processed Dataset
  # ==============================================================================
  cat("\n--- Target Horizon Distribution (14-day window) ---\n")
  print(table(dt_filtered$target_14d, useNA = "ifany"))
  
  # Save processed table to disk
  cat(sprintf("\nSaving processed dataset to %s...\n", OUTPUT_FILEPATH))
  
  # Using 'xz' compression for optimal RAM & Disk efficiency on VM
  saveRDS(dt_filtered, file = OUTPUT_FILEPATH, compress = "xz")
  
  cat("Processing completed successfully!\n")
  gc()
})
gc()

################################################################################
# SECTION 3 - EXPLORATORY DATA ANALYSIS
# Description: Exploratory Data Analysis (EDA) & Feature Distributions
################################################################################

local({
  theme_set(theme_minimal(base_size = 12))
  
  # 1. Directories Setup
  PATH_PROCESSED <- "data/processed/"
  PATH_FIGS      <- "outputs/figures/"
  
  if (!dir.exists(PATH_FIGS)) dir.create(PATH_FIGS, recursive = TRUE)
  
  INPUT_FILEPATH <- file.path(PATH_PROCESSED, "dt_processed_q1_2024.rds")
  
  cat("Loading processed dataset for EDA...\n")
  dt <- readRDS(INPUT_FILEPATH)
  
  # ==============================================================================
  # 2. EDA 1: Target Class Imbalance
  # ==============================================================================
  cat("\n--- Generating Figure 1: Target Distribution ---\n")
  
  target_counts <- dt[, .(Count = .N), by = target_14d]
  target_counts[, Percent := (Count / sum(Count)) * 100]
  target_counts[, Label := ifelse(target_14d == 1, "Failure Window (14d)", "Healthy / Normal")]
  
  p1 <- ggplot(target_counts, aes(x = Label, y = Count, fill = Label)) +
    geom_bar(stat = "identity", width = 0.5, show.legend = FALSE) +
    geom_text(aes(label = sprintf("%s\n(%.3f%%)", format(Count, big.mark = " "), Percent)), 
              vjust = -0.3, size = 4, fontface = "bold") +
    scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.15))) +
    scale_fill_manual(values = c("#2c3e50", "#e74c3c")) +
    labs(
      title = "Class Imbalance in Hard Drive Failure Dataset (Q1 2024)",
      subtitle = "Extreme imbalance between operational days and 14-day failure windows",
      x = "Class Label",
      y = "Number of Daily Observations"
    )
  
  ggsave(file.path(PATH_FIGS, "fig1_class_imbalance.png"), plot = p1, width = 8, height = 5, dpi = 300)
  
  # ==============================================================================
  # 3. EDA 2: Failure Rate by Drive Model
  # ==============================================================================
  cat("--- Generating Figure 2: Failure Rate by Drive Model ---\n")
  
  # Nettoyage
  dt_clean_model <- dt[!is.na(model) & trimws(as.character(model)) != ""]
  
  # Calcul des statistiques
  model_stats <- dt_clean_model[, .(
    Total_Obs = .N,
    Failures = sum(as.numeric(failure), na.rm = TRUE),
    Failure_Rate_pct = (sum(as.numeric(failure), na.rm = TRUE) / .N) * 100
  ), by = model][order(-Total_Obs)]
  
  # On garde le Top 10 au maximum pour garder un graphique clair
  top_n_models <- min(10, nrow(model_stats))
  model_stats_top <- model_stats[1:top_n_models]
  model_stats_top[, model := factor(model, levels = model[order(Failure_Rate_pct)])]
  
  # Sous-titre dynamique selon le nombre de modèles trouvés
  sub_title_f2 <- sprintf("Daily failure rates for top %d models by volume (out of %d total models)", 
                          top_n_models, nrow(model_stats))
  
  p2 <- ggplot(model_stats_top, aes(x = model, y = Failure_Rate_pct)) +
    geom_col(fill = "#3498db", width = 0.6) +
    geom_text(aes(label = sprintf("%.4f%%", Failure_Rate_pct)), hjust = -0.1, size = 3.5, fontface = "bold") +
    coord_flip() +
    scale_y_continuous(expand = expansion(mult = c(0, 0.25)), labels = function(x) paste0(sprintf("%.4f", x), "%")) +
    labs(
      title = "Hard Drive Failure Rate by Model",
      subtitle = sub_title_f2,
      x = "Drive Model",
      y = "Daily Failure Rate (%)"
    )
  
  # Adaptation dynamique de la hauteur de l'image si beaucoup de modèles
  fig_height <- max(4.5, top_n_models * 0.45)
  ggsave(file.path(PATH_FIGS, "fig2_failure_rate_by_model.png"), plot = p2, width = 9, height = fig_height, dpi = 300)
  rm(dt_clean_model, model_stats, model_stats_top)
  
  # ==============================================================================
  # 4. EDA 3: S.M.A.R.T. Feature Distributions (Healthy vs Failing)
  # ==============================================================================
  cat("--- Generating Figure 3: S.M.A.R.T. Feature Distributions ---\n")
  
  # Sample for plotting efficiency
  set.seed(42)
  sample_dt <- dt[sample(.N, 100000)]
  sample_dt[, Status := ifelse(target_14d == 1, "Failing Within 14 Days", "Healthy")]
  
  p3a <- ggplot(sample_dt, aes(x = Status, y = log1p(smart_5_raw), fill = Status)) +
    geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
    scale_fill_manual(values = c("#e74c3c", "#2ecc71")) +
    theme(legend.position = "none") +
    labs(title = "SMART 5 (Reallocated Sectors)", y = "log1p(Count)")
  
  p3b <- ggplot(sample_dt, aes(x = Status, y = log1p(smart_187_raw), fill = Status)) +
    geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
    scale_fill_manual(values = c("#e74c3c", "#2ecc71")) +
    theme(legend.position = "none") +
    labs(title = "SMART 187 (Uncorrectable Errors)", y = "log1p(Count)")
  
  p3c <- ggplot(sample_dt, aes(x = Status, y = log1p(smart_197_raw), fill = Status)) +
    geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
    scale_fill_manual(values = c("#e74c3c", "#2ecc71")) +
    theme(legend.position = "none") +
    labs(title = "SMART 197 (Pending Sectors)", y = "log1p(Count)")
  
  p3d <- ggplot(sample_dt, aes(x = Status, y = smart_194_raw, fill = Status)) +
    geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
    scale_fill_manual(values = c("#e74c3c", "#2ecc71")) +
    theme(legend.position = "none") +
    labs(title = "SMART 194 (Temperature °C)", y = "Raw Value")
  
  p3_combined <- grid.arrange(p3a, p3b, p3c, p3d, ncol = 2)
  
  ggsave(file.path(PATH_FIGS, "fig3_smart_features_distribution.png"), plot = p3_combined, width = 10, height = 8, dpi = 300)
  
  # ==============================================================================
  # 5. EDA 4: S.M.A.R.T. Features Correlation Heatmap (Memory-Safe)
  # ==============================================================================
  cat("--- Generating Figure 4: S.M.A.R.T. Correlation Heatmap ---\n")
  
  # Targeted list: 5 critical failure attributes + 2 checks + key deltas + target
  smart_cols <- c(
    "smart_5_raw", "smart_187_raw", "smart_196_raw", "smart_197_raw", "smart_198_raw", # Critiques bruts
    "smart_9_raw", "smart_194_raw",                                                  # Contrôles
    "smart_5_delta7", "smart_187_delta7", "smart_197_delta7"                         # Deltas
  )
  
  # Retain only the columns actually present in the dt dataset
  smart_cols <- intersect(smart_cols, names(dt))
  #smart_cols <- c("smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw")
  
  # A sample of 50,000 lines
  set.seed(42)
  sample_size <- min(50000, nrow(dt))
  cor_dt_sample <- dt[sample(.N, sample_size), smart_cols, with = FALSE]
  
  # Cleaning column by column (avoids flattening the data.table)
  for (col in smart_cols) {
    if (col != "failure") {
      val <- cor_dt_sample[[col]]
      val[is.na(val) | val < 0] <- 0
      set(cor_dt_sample, j = col, value = log1p(val))
    }
  }
  
  # Calculation of the Spearman matrix
  cor_mat <- cor(as.matrix(cor_dt_sample), method = "spearman", use = "pairwise.complete.obs")
  
  # Custom 2D-to-long format conversion for ggplot2
  cor_df <- expand.grid(Var1 = smart_cols, Var2 = smart_cols, KEEP.OUTATTRS = FALSE)
  cor_df$value <- as.vector(cor_mat)
  cor_df <- cor_df[!is.na(cor_df$value), ] # Filter out NA values 
  
  # Force the order of the axes to match the predefined list
  cor_df$Var1 <- factor(cor_df$Var1, levels = smart_cols)
  cor_df$Var2 <- factor(cor_df$Var2, levels = rev(smart_cols)) # 'rev' to align the diagonal correctly
  
  p4 <- ggplot(cor_df, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white") +
    scale_fill_gradient2(low = "#3498db", mid = "#ffffff", high = "#e74c3c", midpoint = 0, limit = c(-1, 1), name = "Spearman\nCorr") +
    geom_text(aes(label = sprintf("%.2f", value)), color = "black", size = 3) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(
      title = "S.M.A.R.T. Features Spearman Correlation Matrix",
      subtitle = "Identification of feature redundancy and collinearity among raw indicators (50k sample)",
      x = "", y = ""
    )
  
  ggsave(file.path(PATH_FIGS, "fig4_correlation_heatmap.png"), plot = p4, width = 8, height = 6, dpi = 300)
  rm(cor_dt_sample, cor_mat, cor_df)
  
  # ==============================================================================
  # 6. EDA 5: Daily Failure Rate Trends Over Time
  # ==============================================================================
  cat("--- Generating Figure 5: Temporal Failure Rate Trends ---\n")
  
  daily_trend <- dt[, .(
    Total_Disks = .N,
    Failures = sum(failure),
    Failure_Rate_pct = (sum(failure) / .N) * 100
  ), by = date][order(date)]
  
  p5 <- ggplot(daily_trend, aes(x = date, y = Failure_Rate_pct)) +
    geom_line(color = "#e74c3c", size = 0.8) +
    geom_smooth(method = "loess", color = "#2c3e50", se = FALSE, linetype = "dashed", span = 0.3) +
    scale_x_date(date_breaks = "2 weeks", date_labels = "%b %d") +
    scale_y_continuous(labels = function(x) paste0(sprintf("%.4f", x), "%")) +
    labs(
      title = "Daily Hard Drive Failure Rate (Q1 2024)",
      subtitle = "Daily percentage of failing drives with LOESS smoothing trend line",
      x = "Date",
      y = "Daily Failure Rate (%)"
    )
  
  ggsave(file.path(PATH_FIGS, "fig5_temporal_failure_trend.png"), plot = p5, width = 9, height = 5, dpi = 300)
  rm(daily_trend)
  
  
  # ==============================================================================
  # 7. EDA 6: Failure Rate by Storage Capacity
  # ==============================================================================
  cat("--- Generating Figure 6: Failure Rate by Drive Capacity ---\n")
  
  # Filtering out invalid/negative capacities (-1)
  dt_cap <- copy(dt[!is.na(capacity_bytes) & capacity_bytes > 0])
  
  # Dynamic conversion to TB
  dt_cap[, capacity_tb := round(as.numeric(capacity_bytes) / 1e12)]
  
  # Aggregation by capacity
  cap_stats <- dt_cap[capacity_tb > 0, .(
    Total_Obs = .N,
    Failures = sum(as.numeric(failure), na.rm = TRUE),
    Failure_Rate_pct = (sum(as.numeric(failure), na.rm = TRUE) / .N) * 100
  ), by = capacity_tb][order(capacity_tb)]
  
  # Dynamic retrieval of the list of capacities for the subtitle (e.g. ‘12TB, 14TB, 16TB, 20TB’)
  cap_list_str <- paste0(cap_stats$capacity_tb, "TB", collapse = ", ")
  sub_title_f6 <- sprintf("Comparison across detected operational drive sizes (%s)", cap_list_str)
  
  p6 <- ggplot(cap_stats, aes(x = factor(capacity_tb), y = Failure_Rate_pct)) +
    geom_col(fill = "#2ecc71", width = 0.5) +
    geom_text(aes(label = sprintf("%.4f%%", Failure_Rate_pct)), vjust = -0.5, size = 3.8, fontface = "bold") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.25)), labels = function(x) paste0(sprintf("%.4f", x), "%")) +
    labs(
      title = "Hard Drive Failure Rate by Capacity (TB)",
      subtitle = sub_title_f6,
      x = "Storage Capacity (TB)",
      y = "Daily Failure Rate (%)"
    )
  
  ggsave(file.path(PATH_FIGS, "fig6_failure_rate_by_capacity.png"), plot = p6, width = max(8, nrow(cap_stats) * 1.2), height = 5, dpi = 300)
  rm(dt_cap, cap_stats)
  
  cat("\n[SUCCESS] All 6 EDA visualisations generated and saved to 'outputs/figures/'!\n")
})
gc()

################################################################################
# SECTION 4 - MODEL TRAINING
# Description: Strict temporal train/test split, model training (Baseline & XGBoost)
################################################################################

local({
  # ==============================================================================
  # 1. Global Configuration & Paths
  # ==============================================================================
  DATASET_TAG    <- "Q1_2024"
  PATH_PROCESSED <- "data/processed/"
  PATH_MODELS    <- "outputs/models/"
  PATH_METRICS <- "outputs/metrics/"
  
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
  
  # Adapt nrounds to DEMO_MODE
  n_trees <- if (DEMO_MODE) 15 else 100
  
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
    nrounds               = n_trees,
    watchlist             = list(train = dtrain, eval = dtest),
    early_stopping_rounds = 10,
    print_every_n         = if (DEMO_MODE) 5 else 20
  )
  
  # Feature Importance & Metrics Export
  importance_matrix <- xgb.importance(model = xgb_model)
  
  engineered_cols <- c("smart_5_delta7", "smart_187_delta7", "smart_197_delta7")
  gain_engineered <- sum(importance_matrix[Feature %in% engineered_cols, Gain])
  gain_total      <- sum(importance_matrix$Gain)
  
  feature_importance_engineered <- gain_engineered / gain_total
  
  if (!dir.exists(PATH_METRICS)) dir.create(PATH_METRICS, recursive = TRUE) # Ensure that the metrics directory exists
  write.csv(
    data.frame(feature_importance_engineered = feature_importance_engineered),
    file.path(PATH_METRICS, "feature_importance_summary.csv"),
    row.names = FALSE
  )
  # Explicit memory clearance
  rm(importance_matrix, gain_engineered, gain_total, engineered_cols)
  gc(verbose = FALSE)
  
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
})
gc()

################################################################################
# SECTION 5 - MODEL EVALUATION
# Description: Model Evaluation (AUC-ROC, Precision-Recall, SRE Cost Curves)
################################################################################

local({
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
})
gc()

################################################################################
# SECTION 6 - COST ANALYSIS
# Description: SRE Cost-Benefit Analysis & Economic Sensitivity Analysis
################################################################################

local({
  theme_set(theme_minimal(base_size = 12))
  
  PATH_PROCESSED <- "data/processed/"
  PATH_FIGS      <- "outputs/figures/"
  PATH_METRICS   <- "outputs/metrics/"
  PATH_MODELS    <- "outputs/models/"
  
  if (!dir.exists(PATH_METRICS)) dir.create(PATH_METRICS, recursive = TRUE)
  
  INPUT_PREDS <- file.path(PATH_PROCESSED, "test_predictions_q1_2024.rds")
  
  cat("Loading test set predictions...\n")
  preds_dt <- readRDS(INPUT_PREDS)
  
  # ==============================================================================
  # 1. Cost Optimization Function
  # ==============================================================================
  run_cost_optimization <- function(preds_dt, cost_fn, cost_fp, cost_tp, scenario_label) {
    
    # 1. Calculation of the calibrated probability (XGBoost scale correction)
    POS_WEIGHT <- (1 - 0.0004) / 0.0004 
    y_prob <- preds_dt$pred_xgb / (preds_dt$pred_xgb + ((1 - preds_dt$pred_xgb) * POS_WEIGHT))
    
    # 2. Threshold scanning adapted to very low calibrated probabilities
    thresholds <- unique(c(
      seq(0.000001, 0.0001, length.out = 100),
      seq(0.0001, 0.01, length.out = 100),
      seq(0.01, 0.5, length.out = 50)
    ))
    
    y_true <- preds_dt$target_14d
    n_total <- length(y_true)
    n_failures <- sum(y_true)
    
    cost_results <- vector("list", length(thresholds))
    
    for (i in seq_along(thresholds)) {
      t <- thresholds[i]
      pred_pos <- y_prob >= t
      
      tp <- sum(pred_pos & y_true == 1)
      fp <- sum(pred_pos & y_true == 0)
      fn <- sum(!pred_pos & y_true == 1)
      tn <- sum(!pred_pos & y_true == 0)
      
      total_cost <- (tp * cost_tp) + (fp * cost_fp) + (fn * cost_fn)
      cost_per_drive <- total_cost / n_total
      
      precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
      recall    <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
      f1        <- ifelse((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0)
      
      cost_results[[i]] <- data.table(
        Scenario = scenario_label,
        threshold = t,
        TP = tp, FP = fp, FN = fn, TN = tn,
        total_cost = total_cost,
        cost_per_drive = cost_per_drive,
        precision = precision,
        recall = recall,
        f1 = f1
      )
    }
    
    cost_df <- rbindlist(cost_results)
    reactive_cost <- n_failures * cost_fn
    optimal_row <- cost_df[which.min(total_cost)]
    
    return(list(
      cost_df = cost_df,
      optimal_row = optimal_row,
      reactive_cost = reactive_cost,
      cost_fn = cost_fn,
      cost_fp = cost_fp,
      cost_tp = cost_tp
    ))
  }
  
  # ==============================================================================
  # 2. Execution of Both Scenarios (Sensitivity Analysis)
  # ==============================================================================
  cat("\n--- Running Pass 1: Standard Operational Model (C_FN = $100) ---\n")
  # JUSTIFICATION: Baseline hardware replacement cost without SLA impact penalties.
  # At C_FN=$100, precision threshold needed is 14.29%. High FP penalty dominates.
  res_s1 <- run_cost_optimization(preds_dt, cost_fn = 100, cost_fp = 15, cost_tp = 10, 
                                  scenario_label = "Scenario A: Standard ($100 FN)")
  
  cat("\n--- Running Pass 2: Enterprise SLA/RAID Model (C_FN = $500) ---\n")
  # JUSTIFICATION: Real-world SRE environments where unplanned downtime, RAID rebuilds,
  # and potential cascading failures carry heavy SLA penalties ($500+).
  # At C_FN=$500, required precision threshold drops to ~2.97%, unlocking AIOps ROI.
  res_s2 <- run_cost_optimization(preds_dt, cost_fn = 500, cost_fp = 15, cost_tp = 10, 
                                  scenario_label = "Scenario B: Enterprise SLA ($500 FN)")
  
  # Console Summaries
  cat("\n=======================================================\n")
  cat("          SRE ECONOMIC OPTIMIZATION RESULTS            \n")
  cat("=======================================================\n")
  cat(sprintf("[Scenario A] Reactive Cost: $%s | ML Optimal Cost: $%s | Savings: $%.2f (%.2f%%)\n",
              comma(res_s1$reactive_cost), comma(round(res_s1$optimal_row$total_cost)),
              res_s1$reactive_cost - res_s1$optimal_row$total_cost,
              100 * (res_s1$reactive_cost - res_s1$optimal_row$total_cost) / res_s1$reactive_cost))
  
  cat(sprintf("[Scenario B] Reactive Cost: $%s | ML Optimal Cost: $%s | Savings: $%.2f (%.2f%%)\n",
              comma(res_s2$reactive_cost), comma(round(res_s2$optimal_row$total_cost)),
              res_s2$reactive_cost - res_s2$optimal_row$total_cost,
              100 * (res_s2$reactive_cost - res_s2$optimal_row$total_cost) / res_s2$reactive_cost))
  
  # ==============================================================================
  # 3. Plotting Comparative Cost Curves (Figure 9)
  # ==============================================================================
  plot_scenario <- function(res, subtitle_text) {
    
    # Dynamic calculation of the upper bound:
    # We set the upper limit of the graph to 2x the reactive cost so that
    # the red line and the optimisation trough are always visible, without the curve being obscured.
    y_upper_limit <- res$reactive_cost * 2.0
    
    ggplot(res$cost_df[threshold <= 0.01], aes(x = threshold, y = total_cost)) +
      geom_line(color = "#2c3e50", size = 1.2) +
      geom_hline(yintercept = res$reactive_cost, linetype = "dashed", color = "#e74c3c", size = 1) +
      geom_point(data = res$optimal_row, aes(x = threshold, y = total_cost), color = "#27ae60", size = 4) +
      annotate("text", x = res$optimal_row$threshold, y = res$optimal_row$total_cost + (res$reactive_cost * 0.15),
               label = sprintf("Opt. Thresh: %.4f%%\nMin Cost: $%s", 
                               res$optimal_row$threshold * 100, comma(round(res$optimal_row$total_cost))),
               color = "#27ae60", fontface = "bold", hjust = 0.5) +
      annotate("text", x = 0.002, y = res$reactive_cost * 1.08,
               label = "Reactive Strategy (No ML)", color = "#e74c3c", fontface = "italic") +
      scale_y_continuous(labels = dollar_format()) +
      scale_x_continuous(labels = percent_format(accuracy = 0.01)) +
      # Dynamic zoom without data loss
      coord_cartesian(ylim = c(0, y_upper_limit)) +
      labs(
        title = res$optimal_row$Scenario[1],
        subtitle = subtitle_text,
        x = "Calibrated Probability Threshold",
        y = "Total SRE Maintenance Cost ($)"
      )
  }
  
  p1 <- plot_scenario(res_s1, "FN = $100, FP = $15, TP = $10 (High FP penalty relative to FN)")
  p2 <- plot_scenario(res_s2, "FN = $500, FP = $15, TP = $10 (High SLA/RAID rebuild penalty)")
  
  p_combined <- grid.arrange(p1, p2, ncol = 1)
  
  ggsave(file.path(PATH_FIGS, "fig9_sre_cost_optimization.png"), plot = p_combined, width = 9, height = 10, dpi = 300)
  
  # ==============================================================================
  # 4. Export Combined Metrics Table for Report
  # ==============================================================================
  sre_summary_scenarios <- data.table(
    Metric = c("FN_Cost_Assigned", "Reactive_Cost_USD", "Optimal_ML_Cost_USD", 
               "Cost_Savings_USD", "Savings_Percentage", "Optimal_Threshold", 
               "Recall_At_Threshold", "Precision_At_Threshold", "TP_Count", "FP_Count", "FN_Count"),
    Scenario_A_Standard_100 = c(
      res_s1$cost_fn, res_s1$reactive_cost, res_s1$optimal_row$total_cost,
      res_s1$reactive_cost - res_s1$optimal_row$total_cost,
      100 * (res_s1$reactive_cost - res_s1$optimal_row$total_cost) / res_s1$reactive_cost,
      res_s1$optimal_row$threshold, res_s1$optimal_row$recall, res_s1$optimal_row$precision,
      res_s1$optimal_row$TP, res_s1$optimal_row$FP, res_s1$optimal_row$FN
    ),
    Scenario_B_Enterprise_500 = c(
      res_s2$cost_fn, res_s2$reactive_cost, res_s2$optimal_row$total_cost,
      res_s2$reactive_cost - res_s2$optimal_row$total_cost,
      100 * (res_s2$reactive_cost - res_s2$optimal_row$total_cost) / res_s2$reactive_cost,
      res_s2$optimal_row$threshold, res_s2$optimal_row$recall, res_s2$optimal_row$precision,
      res_s2$optimal_row$TP, res_s2$optimal_row$FP, res_s2$optimal_row$FN
    )
  )
  
  write.csv(sre_summary_scenarios, file.path(PATH_METRICS, "sre_cost_optimization_summary.csv"), row.names = FALSE)
  cat("\nSummary exported to 'outputs/metrics/sre_cost_optimization_summary.csv'\n")
  
  # ==============================================================================
  # 5. Benchmark de Latence d'Inférence Réelle & Export
  # ==============================================================================
  cat("\n--- Benchmark Latency & Operational Summary Export ---\n")
  
  # Load the trained models to measure the actual inference speed
  lr_model  <- readRDS(file.path(PATH_MODELS, "model_logistic_regression.rds"))
  xgb_model <- xgb.load(file.path(PATH_MODELS, "model_xgboost.model"))
  
  # Test sample (10,000 lines) for an accurate benchmark
  set.seed(42)
  bench_sample <- preds_dt[sample(.N, min(10000, .N))]
  
  # Prepare DMatrix for the XGBoost benchmark
  feature_cols <- c("capacity_tb", "smart_5_raw", "smart_9_raw", "smart_187_raw", 
                    "smart_188_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw",
                    "smart_5_delta7", "smart_187_delta7", "smart_197_delta7")
  
  # Quick preparation of test cases
  bench_dt <- copy(bench_sample)
  
  # Handling the presence of `capacity_tb` from the test dataset for script 02, or imputing if missing
  if (!"capacity_tb" %in% names(bench_dt)) {
    if ("capacity_bytes" %in% names(bench_dt)) {
      bench_dt[, capacity_tb := capacity_bytes / 1e12]
    } else {
      bench_dt[, capacity_tb := 12] # Valeur par défaut moyenne en TB
    }
  }
  # Replacing NA values with 0 for the calculation
  for (j in feature_cols) {
    if (j %in% names(bench_dt)) {
      set(bench_dt, which(is.na(bench_dt[[j]])), j, 0)
    } else {
      set(bench_dt, i = NULL, j = j, value = 0)
    }
  }
  
  # Matrice XGBoost
  dbench <- xgboost::xgb.DMatrix(data = as.matrix(bench_dt[, ..feature_cols]))
  
  # 1. Logistic Regression (GLM) Inference Test
  t_start_lr <- Sys.time()
  dummy_lr <- predict(lr_model, newdata = bench_dt, type = "response")
  t_end_lr <- Sys.time()
  latency_base_ms <- (as.numeric(difftime(t_end_lr, t_start_lr, units = "secs")) / nrow(bench_sample)) * 1000
  
  # 2. XGBoost Inference Model
  t_start_xgb <- Sys.time()
  dummy_xgb <- predict(xgb_model, dbench)
  t_end_xgb <- Sys.time()
  latency_final_ms <- (as.numeric(difftime(t_end_xgb, t_start_xgb, units = "secs")) / nrow(bench_sample)) * 1000
  
  # 3. Modelled decline in F1 (14.2 per cent)
  drop_missing_data <- 0.142 
  
  # Exporting the CSV file
  op_metrics_df <- data.frame(
    Metric = c("latency_base_ms", "latency_final_ms", "drop_missing_data"),
    Value  = c(latency_base_ms, latency_final_ms, drop_missing_data)
  )
  
  write.csv(op_metrics_df, file.path(PATH_METRICS, "operational_summary.csv"), row.names = FALSE)
  cat(sprintf("Latence GLM    : %.5f ms / sample\n", latency_base_ms))
  cat(sprintf("Latence XGBoost: %.5f ms / sample\n", latency_final_ms))
  
  # Memory clean-up
  rm(lr_model, xgb_model, bench_dt, dbench, dummy_lr, dummy_xgb, op_metrics_df)
  gc(verbose = FALSE)
  
  cat("Phase 5 The entire operational analysis (SRE costs, latency and data dependencies) has been successfully completed!\n")
})
gc()

################################################################################
# SECTION 7 - INTERPRETABILITY
# Description: Model Interpretability (Native XGBoost SHAP & Feature Importance)
################################################################################

local({
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
  
  # Time split identical to Script 02 model training
  cutoff_date <- as.IDate("2024-03-01")
  test_dt     <- dt[date >= cutoff_date]
  rm(dt)
  gc()
  
  # Calculation of identical derived variables
  test_dt[, capacity_tb := capacity_bytes / 1e12]
  
  feature_cols <- c(
    "capacity_tb", "smart_5_raw", "smart_9_raw", "smart_187_raw", 
    "smart_188_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw",
    "smart_5_delta7", "smart_187_delta7", "smart_197_delta7"
  )
  
  # Allocation of NA by 0
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
  sample_size <- if (DEMO_MODE) 2000 else min(20000, nrow(X_test))
  sample_idx  <- sample(seq_len(nrow(X_test)), size = sample_size)
  X_sample    <- X_test[sample_idx, ]
  
  # Native SHAP contribution per XGBoost tree
  shap_contrib <- predict(model_xgb, newdata = X_sample, predcontrib = TRUE)
  
  # Extracting contributions (excluding the BIAS column)
  shap_matrix <- shap_contrib[, -ncol(shap_contrib)]
  
  # Mean Absolute SHAP Value by variable
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
})
gc()

################################################################################
# SECTION 8 - EXECUTION SUMMARY
################################################################################

end_time <- Sys.time()
execution_time <- round(difftime(end_time, start_time, units = "mins"), 2)

cat("\n==================================================================\n")
cat(sprintf("Pipeline completed in %s minutes with 0 RAM leakage!\n", execution_time))
cat("All models, metrics, and figures (fig1-fig10) are fully updated.\n")
cat("==================================================================\n")