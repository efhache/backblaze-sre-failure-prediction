# ==============================================================================
# Script: 01_eda_and_processing.R
# Project: Predictive Hard Drive Failure Modelling for SRE Operations
# Description: Optimised data ingestion, exploration (EDA) and data preparation
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

cat("=========================================================\n")
cat(" Environment successfully initialised for the VM (6 GB RAM)\n")
cat("=========================================================\n")

# TODO:
# - Import the Backblaze dataset using data.table::fread()
# - Filter for representative models
# - Feature engineering (Y_it horizon [14–30 days])
# - EDA summary & export of the .rds file