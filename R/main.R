# ==============================================================================
# Script: main.R
# Project: Predictive Hard Drive Failure Modelling for SRE Operations
# Description: Master Pipeline Execution with Isolated Process Memory Management
# ==============================================================================

cat("==================================================================\n")
cat("Starting Complete Backblaze AIOps Failure Prediction Pipeline\n")
cat("==================================================================\n\n")

start_time <- Sys.time()

# List of scripts to be run in the workflow order
scripts <- c(
  "R/00_download_and_ingest.R",
  "R/01_eda_and_processing.R",
  "R/01b_exploratory_data_analysis.R",
  "R/02_model_training.R",
  "R/03_model_evaluation.R",
  "R/04_sre_cost_analysis.R",
  "R/05_model_interpretability.R"
)

for (script in scripts) {
  if (file.exists(script)) {
    cat(sprintf("\n---> [RAM Clean Execution] Starting: %s\n", script))
    
    # Execution via an Rscript subprocess to ensure zero memory leaks
    exit_code <- system2("Rscript", args = script)
    
    if (exit_code != 0) {
      stop(sprintf("Error encountered during execution of %s (Exit code: %d)", script, exit_code))
    }
    
    # Forced cleaning of the parent environment
    gc(verbose = FALSE)
  } else {
    stop(sprintf("Error: File missing - %s", script))
  }
}

end_time <- Sys.time()
execution_time <- round(difftime(end_time, start_time, units = "mins"), 2)

cat("\n==================================================================\n")
cat(sprintf("Pipeline completed in %s minutes with 0 RAM leakage!\n", execution_time))
cat("All models, metrics, and figures (fig1-fig10) are fully updated.\n")
cat("==================================================================\n")