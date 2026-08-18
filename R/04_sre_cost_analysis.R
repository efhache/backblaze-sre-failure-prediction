# ==============================================================================
# Script: 04_sre_cost_analysis.R
# Project: Predictive Hard Drive Failure Modelling for SRE Operations
# Description: SRE Cost-Benefit Analysis & Economic Sensitivity Analysis
# ==============================================================================

# 0. Load Libraries
required_packages <- c("data.table", "tidyverse", "scales", "gridExtra")

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

ggsave(file.path(PATH_FIGS, "fig9_sre_cost_optimization.png"), plot = p_complete <- grid.arrange(p1, p2, ncol = 1), width = 9, height = 10, dpi = 300)

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
cat("Phase 5 SRE Cost Sensitivity Analysis completed successfully!\n")