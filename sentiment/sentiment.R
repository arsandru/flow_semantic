library(jsonlite)
library(dplyr)
library(tidyr)
library(performance)
library(ggplot2)
library(emmeans)
library(clubSandwich)

df <- read.csv("sentiment/vader_sentiment_emotional_only.csv", check.names = FALSE)

# Drop NA condition and align IDs
data <- df %>%
  filter(!is.na(Exp_Condition)) %>%
  mutate(ID = as.character(ID))

# Parse VADER score dicts for neg/neu/pos
# Keep compound from the CSV to avoid duplicate-name issues
data$scores_json <- gsub("'", "\"", data$scores)
parsed <- lapply(data$scores_json, fromJSON)
scores_df <- bind_rows(parsed) %>% dplyr::select(neg, neu, pos)
data <- bind_cols(data, scores_df)

# -----------------------------
# A) Label model: Score ~ Exp_Condition * Label
# -----------------------------
data_long <- data %>%
  select(ID, Exp_Condition, neg, neu, pos) %>%
  pivot_longer(
    cols = c(neg, neu, pos),
    names_to = "Label",
    values_to = "Score"
  ) %>%
  mutate(
    Label = as.factor(Label),
    Exp_Condition = factor(as.character(Exp_Condition), levels = c("3", "1", "2"))
  )

model <- lm(Score ~ Exp_Condition * Label, data = data_long)

# Diagnostics (same pattern as word_level.R)
diag_summary <- summary(model)
diag_hetero <- check_heteroscedasticity(model)
diag_normality <- check_normality(model)
diag_outliers <- check_outliers(model)
diag_perf <- model_performance(model)

print(diag_summary)
print(diag_hetero)
print(diag_normality)
print(diag_outliers)
print(diag_perf)

# CR2 robust inference clustered by participant
V_score <- vcovCR(model, cluster = data_long$ID, type = "CR2")
coef_tab_score <- coef_test(model, vcov = V_score)
print(coef_tab_score)

# Residual plots
diag_df <- data.frame(fitted = fitted(model), resid = resid(model))
resid_fitted_plot <- ggplot(diag_df, aes(x = fitted, y = resid)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs Fitted", x = "Fitted values", y = "Residuals") +
  theme_minimal()

qq_plot <- ggplot(diag_df, aes(sample = resid)) +
  stat_qq() +
  stat_qq_line(color = "red") +
  labs(title = "Q-Q Plot of Residuals", x = "Theoretical quantiles", y = "Sample quantiles") +
  theme_minimal()

print(resid_fitted_plot)
print(qq_plot)

ggsave("sentiment/residuals_vs_fitted_sentiment.pdf", plot = resid_fitted_plot, width = 8, height = 6)
ggsave("sentiment/qq_plot_sentiment.pdf", plot = qq_plot, width = 8, height = 6)

# Save diagnostics report
writeLines(c(
  "Sentiment model diagnostics report",
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "summary(model):", capture.output(diag_summary),
  "",
  "check_heteroscedasticity(model):", capture.output(diag_hetero),
  "",
  "check_normality(model):", capture.output(diag_normality),
  "",
  "check_outliers(model):", capture.output(diag_outliers),
  "",
  "model_performance(model):", capture.output(diag_perf),
  "",
  "coef_test(model, CR2):", capture.output(coef_tab_score)
), "sentiment/diagnostics_report_sentiment.txt")

# Plot means + 95% CI
means_df <- data_long %>%
  group_by(Exp_Condition, Label) %>%
  summarise(
    mean = mean(Score, na.rm = TRUE),
    SE = sd(Score, na.rm = TRUE) / sqrt(sum(!is.na(Score))),
    n = sum(!is.na(Score)),
    .groups = "drop"
  ) %>%
  mutate(
    t_crit = qt(0.975, df = pmax(n - 1, 1)),
    lower = mean - t_crit * SE,
    upper = mean + t_crit * SE,
    Exp_Condition = factor(as.character(Exp_Condition), levels = c("3", "1", "2"))
  )

# Pairwise within label, using CR2 vcov
emm <- emmeans(model, ~ Exp_Condition | Label, vcov. = V_score)
pw <- as.data.frame(pairs(emm))

split_contrast <- strsplit(as.character(pw$contrast), " - ", fixed = TRUE)
pw$x1 <- sapply(split_contrast, `[`, 1)
pw$x2 <- sapply(split_contrast, `[`, 2)
pw$label <- ifelse(pw$p.value < 0.001, "***",
            ifelse(pw$p.value < 0.01, "**",
            ifelse(pw$p.value < 0.05, "*", NA_character_)))

annot_df <- pw %>%
  filter(!is.na(label)) %>%
  mutate(
    Label = as.factor(Label),
    x1 = factor(x1, levels = c("3", "1", "2")),
    x2 = factor(x2, levels = c("3", "1", "2"))
  )

if (nrow(annot_df) > 0) {
  y_base <- means_df %>%
    group_by(Label) %>%
    summarise(y_max = max(upper, na.rm = TRUE), .groups = "drop")

  annot_df <- annot_df %>%
    left_join(y_base, by = "Label") %>%
    group_by(Label) %>%
    mutate(y = y_max + row_number() * 0.05) %>%
    ungroup()
}

cond_colors <- c("1" = "#8de5a1", "2" = "#ffb482", "3" = "#a1c9f4")

p <- ggplot(means_df, aes(x = Exp_Condition, y = mean, color = Exp_Condition)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15) +
  facet_wrap(~ Label, scales = "free_y") +
  scale_color_manual(
    values = cond_colors,
    breaks = c("1", "2", "3"),
    labels = c("Flow (VR+Meditation)", "VR Only", "Control (Care as Usual)")
  ) +
  scale_x_discrete(breaks = c("3", "1", "2"), labels = c("Control", "Flow", "VR Only")) +
  labs(
    x = "Condition",
    y = "Mean sentiment score",
    title = "Sentiment means by Condition and Label",
    subtitle = "Pairwise significance from CR2-robust tests; error bars are 95% CI"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

if (nrow(annot_df) > 0) {
  p <- p +
    geom_segment(data = annot_df, aes(x = x1, xend = x2, y = y, yend = y), inherit.aes = FALSE, color = "black") +
    geom_text(
      data = annot_df,
      aes(
        x = (as.numeric(factor(x1, levels = c("3", "1", "2"))) +
             as.numeric(factor(x2, levels = c("3", "1", "2")))) / 2,
        y = y + 0.02,
        label = label
      ),
      inherit.aes = FALSE,
      size = 5,
      color = "black"
    )
}

print(p)
ggsave("sentiment/sentiment_condition_label_plot.pdf", plot = p, width = 12, height = 7)

# Save CR2 outputs
write.csv(as.data.frame(coef_tab_score), "sentiment/coefficients_sentiment_cr2.csv", row.names = FALSE)
write.csv(pw, "sentiment/pairwise_sentiment_cr2.csv", row.names = FALSE)
write.csv(annot_df, "sentiment/significant_pairwise_findings_sentiment.csv", row.names = FALSE)

# -----------------------------
# B) Compound model: compound ~ Exp_Condition
# -----------------------------
data_compound <- data %>%
  mutate(compound = as.numeric(compound)) %>%
  filter(!is.na(compound)) %>%
  mutate(Exp_Condition = factor(as.character(Exp_Condition), levels = c("3", "1", "2")))

model_compound <- lm(compound ~ Exp_Condition, data = data_compound)
print(summary(model_compound))

# CR2 robust inference
V_compound <- vcovCR(model_compound, cluster = data_compound$ID, type = "CR2")
coef_tab_compound <- coef_test(model_compound, vcov = V_compound)
print(coef_tab_compound)

means_compound <- data_compound %>%
  group_by(Exp_Condition) %>%
  summarise(
    mean = mean(compound, na.rm = TRUE),
    SE = sd(compound, na.rm = TRUE) / sqrt(sum(!is.na(compound))),
    n = sum(!is.na(compound)),
    .groups = "drop"
  ) %>%
  mutate(
    t_crit = qt(0.975, df = pmax(n - 1, 1)),
    lower = mean - t_crit * SE,
    upper = mean + t_crit * SE
  )

emm_compound <- emmeans(model_compound, ~ Exp_Condition, vcov. = V_compound)
pw_compound <- as.data.frame(pairs(emm_compound))

split_comp <- strsplit(as.character(pw_compound$contrast), " - ", fixed = TRUE)
pw_compound$x1 <- sapply(split_comp, `[`, 1)
pw_compound$x2 <- sapply(split_comp, `[`, 2)
pw_compound$label <- ifelse(pw_compound$p.value < 0.001, "***",
                      ifelse(pw_compound$p.value < 0.01, "**",
                      ifelse(pw_compound$p.value < 0.05, "*", NA_character_)))

annot_compound <- pw_compound %>%
  filter(!is.na(label)) %>%
  mutate(
    x1 = factor(x1, levels = c("3", "1", "2")),
    x2 = factor(x2, levels = c("3", "1", "2"))
  )

if (nrow(annot_compound) > 0) {
  y_max_comp <- max(means_compound$upper, na.rm = TRUE)
  annot_compound$y <- y_max_comp + seq_len(nrow(annot_compound)) * 0.05
}

p_compound <- ggplot(means_compound, aes(x = Exp_Condition, y = mean, color = Exp_Condition)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15) +
  scale_color_manual(
    values = cond_colors,
    breaks = c("1", "2", "3"),
    labels = c("Flow (VR+Meditation)", "VR Only", "Control (Care as Usual)")
  ) +
  scale_x_discrete(breaks = c("3", "1", "2"), labels = c("Control", "Flow", "VR Only")) +
  labs(
    x = "Condition",
    y = "Mean compound sentiment",
    title = "Compound Sentiment by Condition",
    subtitle = "Pairwise significance from CR2-robust tests; error bars are 95% CI"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

if (nrow(annot_compound) > 0) {
  p_compound <- p_compound +
    geom_segment(data = annot_compound, aes(x = x1, xend = x2, y = y, yend = y), inherit.aes = FALSE, color = "black") +
    geom_text(
      data = annot_compound,
      aes(
        x = (as.numeric(factor(x1, levels = c("3", "1", "2"))) +
             as.numeric(factor(x2, levels = c("3", "1", "2")))) / 2,
        y = y + 0.02,
        label = label
      ),
      inherit.aes = FALSE,
      size = 5,
      color = "black"
    )
}

print(p_compound)
ggsave("sentiment/sentiment_compound_condition_plot.pdf", plot = p_compound, width = 10, height = 7)

# Save CR2 outputs for compound model
write.csv(as.data.frame(coef_tab_compound), "sentiment/coefficients_compound_cr2.csv", row.names = FALSE)
write.csv(pw_compound, "sentiment/pairwise_compound_cr2.csv", row.names = FALSE)
write.csv(annot_compound, "sentiment/significant_pairwise_findings_sentiment_compound.csv", row.names = FALSE)
