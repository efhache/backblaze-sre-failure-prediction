# ==============================================================================
# Script: 01b_exploratory_data_analysis.R
# Project: Predictive Hard Drive Failure Modelling for SRE Operations
# Description: Exploratory Data Analysis (EDA) & Feature Distributions
# Standard: HarvardX / edX Data Science Capstone Standards
# ==============================================================================

# 0. Load Libraries
required_packages <- c("data.table", "tidyverse", "lubridate", "scales", "gridExtra")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

options(scipen = 999)
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

smart_cols <- c("smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw")

# Échantillonnage de 50 000 lignes
set.seed(42)
sample_size <- min(50000, nrow(dt))
cor_dt_sample <- dt[sample(.N, sample_size), smart_cols, with = FALSE]

# Nettoyage colonne par colonne (évite d'applatir le data.table)
for (col in smart_cols) {
  val <- cor_dt_sample[[col]]
  val[is.na(val) | val < 0] <- 0
  set(cor_dt_sample, j = col, value = log1p(val))
}

# Calcul de la matrice Spearman
cor_mat <- cor(as.matrix(cor_dt_sample), method = "spearman")

# Conversion propre 2D vers format long pour ggplot2
cor_df <- expand.grid(Var1 = smart_cols, Var2 = smart_cols, KEEP.OUTATTRS = FALSE)
cor_df$value <- as.vector(cor_mat)

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

# Filtrage des capacités invalides/négatives (-1)
dt_cap <- copy(dt[!is.na(capacity_bytes) & capacity_bytes > 0])

# Conversion dynamique en TB
dt_cap[, capacity_tb := round(as.numeric(capacity_bytes) / 1e12)]

# Agrégation par capacité
cap_stats <- dt_cap[capacity_tb > 0, .(
  Total_Obs = .N,
  Failures = sum(as.numeric(failure), na.rm = TRUE),
  Failure_Rate_pct = (sum(as.numeric(failure), na.rm = TRUE) / .N) * 100
), by = capacity_tb][order(capacity_tb)]

# Récupération dynamique de la liste des capacités pour le sous-titre (ex: "12TB, 14TB, 16TB, 20TB")
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