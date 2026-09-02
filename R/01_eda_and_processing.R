# ==============================================================================
# Script: 01_eda_and_processing.R
# Project: Predictive Hard Drive Failure Modelling for SRE Operations
# Description: Optimised data ingestion, exploration (EDA) and data preparatio
# ==============================================================================

# 1. Loading the required packages
required_packages <- c("data.table", "tidyverse", "lubridate", "ggplot2")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# 2. Configuring memory and environment options
options(scipen = 999) # Disable scientific notation
gc() # Freeing up RAM (Garbage Collection)

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

library(data.table)
library(tidyverse)
library(lubridate)


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

# Keep top drive models representing majority of the population
top_model_names <- top_10_models[1:5, model]
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