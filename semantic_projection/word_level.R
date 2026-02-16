script_args <- commandArgs(trailingOnly = FALSE)
script_flag <- "--file="
script_path <- sub(script_flag, "", script_args[grep(script_flag, script_args)])
script_dir <- if (length(script_path) > 0) dirname(normalizePath(script_path)) else getwd()

library(lme4)
library(lmerTest)
library(clubSandwich)
library(emmeans)
library(performance)
library(ggplot2)
data <- read.csv(file.path(script_dir, "semantic_projection/semantic_projection_roberta.csv"), stringsAsFactors = FALSE)
data$condition <- factor(data$condition)
data$condition <- relevel(data$condition, ref = "3")

model_words <- lmer(
  projection_fear_vs_calm ~ condition + (1 | Participant),
  data = data
)

# Model fit diagnostics
diag_summary <- summary(model_words)
diag_singularity <- check_singularity(model_words)
diag_convergence <- check_convergence(model_words)
diag_hetero <- check_heteroscedasticity(model_words)
diag_normality <- check_normality(model_words)
diag_outliers <- check_outliers(model_words)
diag_r2 <- r2_nakagawa(model_words)
diag_perf <- model_performance(model_words)

print(diag_summary)
print(diag_singularity)
print(diag_convergence)
print(diag_hetero)
print(diag_normality)
print(diag_outliers)
print(diag_r2)
print(diag_perf)

# Basic residual diagnostic plots
diag_df <- data.frame(
  fitted = fitted(model_words),
  resid = resid(model_words)
)

resid_fitted_plot <- ggplot(diag_df, aes(x = fitted, y = resid)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Residuals vs Fitted",
    x = "Fitted values",
    y = "Residuals"
  ) +
  theme_minimal()

qq_plot <- ggplot(diag_df, aes(sample = resid)) +
  stat_qq() +
  stat_qq_line(color = "red") +
  labs(
    title = "Q-Q Plot of Residuals",
    x = "Theoretical quantiles",
    y = "Sample quantiles"
  ) +
  theme_minimal()

print(resid_fitted_plot)
print(qq_plot)

coef_tab <- coef_test(
  model_words,
  vcov = "CR2"
)
print(coef_tab)

beta_names <- names(fixef(model_words))

V <- vcovCR(
  model_words,
  cluster = model.frame(model_words)$Participant,
  type = "CR2"
)

L_pairwise <- list(
  "1 vs ref" = c(0, 1, 0),
  "2 vs ref" = c(0, 0, 1),
  "1 vs 2"   = c(0, 1, -1)
)

pw_tests <- lapply(L_pairwise, function(L) {
  Wald_test(
    model_words,
    constraints = matrix(L, nrow = 1, dimnames = list(NULL, beta_names)),
    vcov = V,
    test = "chi-sq"
  )
})

# Sanity check: emmeans comparisons with CR2 covariance
emm_cr2 <- emmeans(
  model_words,
  ~ condition,
  vcov. = V
)
print(pairs(emm_cr2))
#print(contrast(emm_cr2, method = "trt.vs.ctrl", ref = "3"))

# Sensitivity analysis: lm + CR2 clustered by Participant
model_lm <- lm(
  projection_fear_vs_calm ~ condition,
  data = data
)

V_lm <- vcovCR(
  model_lm,
  cluster = data$Participant,
  type = "CR2"
)

coef_tab_lm <- coef_test(
  model_lm,
  vcov = V_lm
)
print(coef_tab_lm)

beta_names_lm <- names(coef(model_lm))
pw_tests_lm <- lapply(L_pairwise, function(L) {
  Wald_test(
    model_lm,
    constraints = matrix(L, nrow = 1, dimnames = list(NULL, beta_names_lm)),
    vcov = V_lm,
    test = "chi-sq"
  )
})

comp_df_mixed <- data.frame(
  comparison = names(pw_tests),
  p_mixed_CR2 = sapply(pw_tests, function(x) x$p_val),
  stringsAsFactors = FALSE
)
comp_df_lm <- data.frame(
  comparison = names(pw_tests_lm),
  p_lm_CR2 = sapply(pw_tests_lm, function(x) x$p_val),
  stringsAsFactors = FALSE
)
comparison_check <- merge(comp_df_mixed, comp_df_lm, by = "comparison", all = TRUE)
print(comparison_check)

sig_df <- data.frame(
  comparison = names(pw_tests),
  p_value = sapply(pw_tests, function(x) x$p_val),
  stringsAsFactors = FALSE
)

sig_df$label <- ifelse(sig_df$p_value < 0.001, "***",
                ifelse(sig_df$p_value < 0.01,  "**",
                ifelse(sig_df$p_value < 0.05,  "*", NA_character_)))

sig_df <- subset(sig_df, !is.na(label))

means_df <- aggregate(
  projection_fear_vs_calm ~ condition,
  data = data,
  FUN = function(x) c(
    mean = mean(x, na.rm = TRUE),
    SE = sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))
  )
)

means_df <- do.call(data.frame, means_df)
names(means_df) <- c("condition", "mean", "SE")
# n per condition
means_df <- aggregate(
  projection_fear_vs_calm ~ condition,
  data = data,
  FUN = function(x) c(
    mean = mean(x, na.rm = TRUE),
    SE   = sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x))),
    n    = sum(!is.na(x))
  )
)

means_df <- do.call(data.frame, means_df)
names(means_df) <- c("condition", "mean", "SE", "n")

# exact 95% CI (t-based)
means_df$t_crit <- qt(0.975, df = means_df$n - 1)
means_df$lower  <- means_df$mean - means_df$t_crit * means_df$SE
means_df$upper  <- means_df$mean + means_df$t_crit * means_df$SE


means_df$condition <- factor(as.character(means_df$condition), levels = c("3", "1", "2"))

if (nrow(sig_df) > 0) {
  map_x1 <- c("1 vs ref" = "3", "2 vs ref" = "3", "1 vs 2" = "1")
  map_x2 <- c("1 vs ref" = "1", "2 vs ref" = "2", "1 vs 2" = "2")

  annot_df <- data.frame(
    comparison = sig_df$comparison,
    label = sig_df$label,
    x1 = unname(map_x1[sig_df$comparison]),
    x2 = unname(map_x2[sig_df$comparison]),
    stringsAsFactors = FALSE
  )

  y_max <- max(means_df$upper, na.rm = TRUE)
  annot_df$y <- y_max + seq_len(nrow(annot_df)) * 2
} else {
  annot_df <- data.frame()
}

cond_colors <- c(
  "1" = "#8de5a1",
  "2" = "#ffb482",
  "3" = "#a1c9f4"
)

p <- ggplot(means_df, aes(x = condition, y = mean, color = condition)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15) +
  scale_color_manual(
    values = cond_colors,
    breaks = c("1", "2", "3"),
    labels = c("Flow (VR+Meditation)", "VR Only", "Control (Care as Usual)")
  ) +
  scale_x_discrete(
    breaks = c("3", "1", "2"),
    labels = c("Control", "Flow", "VR Only")
  ) +
  labs(
    x = "Condition",
    y = "Mean fear-calm projection",
    title = "Condition means (CR2-adjusted)",
    subtitle = "Only significant pairwise differences shown; error bars are descriptive mean +/- 1.96*SE",
    color = "Group"
  ) +
  theme_minimal()+
    theme(
        legend.position = "none")

if (nrow(annot_df) > 0) {
  p <- p +
    geom_segment(
      data = annot_df,
      aes(x = x1, xend = x2, y = y, yend = y),
      inherit.aes = FALSE,
      color = "black"
    ) +
    geom_text(
      data = annot_df,
      aes(
        x = (as.numeric(factor(x1, levels = levels(means_df$condition))) +
             as.numeric(factor(x2, levels = levels(means_df$condition)))) / 2,
        y = y + 0.5,
        label = label
      ),
      inherit.aes = FALSE,
      size = 5,
      color = "black"
    )
}

p


ggsave(file.path(script_dir, "semantic_projection_final.pdf"), plot = p, width = 10, height = 8)

# -----------------------------
# 7) Reporting outputs
# -----------------------------
coef_mixed_df <- as.data.frame(coef_tab)
coef_lm_df <- as.data.frame(coef_tab_lm)

write.csv(coef_mixed_df, file.path(script_dir, "coefficients_mixed_cr2.csv"), row.names = FALSE)
write.csv(coef_lm_df, file.path(script_dir, "coefficients_lm_cr2.csv"), row.names = FALSE)
write.csv(comparison_check, file.path(script_dir, "pairwise_comparison_mixed_vs_lm_cr2.csv"), row.names = FALSE)
write.csv(sig_df, file.path(script_dir, "significant_pairwise_findings.csv"), row.names = FALSE)

# Identify significant fixed effects in lm + CR2 table
p_col_lm <- grep("^p", names(coef_lm_df), value = TRUE)[1]
if (!is.na(p_col_lm)) {
  sig_coef_lm <- coef_lm_df[!is.na(coef_lm_df[[p_col_lm]]) & coef_lm_df[[p_col_lm]] < 0.05, , drop = FALSE]
} else {
  sig_coef_lm <- data.frame()
}

# Text report
report_lines <- c(
  "Word-level fear-calm projection analysis report",
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "Saved tables:",
  "- coefficients_mixed_cr2.csv",
  "- coefficients_lm_cr2.csv",
  "- pairwise_comparison_mixed_vs_lm_cr2.csv",
  "- significant_pairwise_findings.csv",
  ""
)

if (nrow(sig_df) == 0) {
  report_lines <- c(report_lines, "Significant pairwise findings (alpha = .05): none", "")
} else {
  report_lines <- c(report_lines, "Significant pairwise findings (alpha = .05):")
  for (i in seq_len(nrow(sig_df))) {
    report_lines <- c(
      report_lines,
      sprintf("- %s: p = %.6f (%s)", sig_df$comparison[i], sig_df$p_value[i], sig_df$label[i])
    )
  }
  report_lines <- c(report_lines, "")
}

if (nrow(sig_coef_lm) == 0) {
  report_lines <- c(report_lines, "Significant LM coefficients (CR2): none")
} else {
  report_lines <- c(report_lines, "Significant LM coefficients (CR2):")
  for (i in seq_len(nrow(sig_coef_lm))) {
    coef_name <- if ("Coef." %in% names(sig_coef_lm)) sig_coef_lm$Coef.[i] else rownames(sig_coef_lm)[i]
    est <- if ("Estimate" %in% names(sig_coef_lm)) sig_coef_lm$Estimate[i] else NA_real_
    pval <- sig_coef_lm[[p_col_lm]][i]
    report_lines <- c(report_lines, sprintf("- %s: estimate = %.6f, p = %.6f", coef_name, est, pval))
  }
}

writeLines(report_lines, file.path(script_dir, "analysis_report.txt"))
cat(paste(report_lines, collapse = "\n"), "\n")



# Save diagnostics report
writeLines(c(
  "Model diagnostics report",
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "summary(model_words):",
  capture.output(diag_summary),
  "",
  "check_singularity(model_words):",
  capture.output(diag_singularity),
  "",
  "check_convergence(model_words):",
  capture.output(diag_convergence),
  "",
  "check_heteroscedasticity(model_words):",
  capture.output(diag_hetero),
  "",
  "check_normality(model_words):",
  capture.output(diag_normality),
  "",
  "check_outliers(model_words):",
  capture.output(diag_outliers),
  "",
  "r2_nakagawa(model_words):",
  capture.output(diag_r2),
  "",
  "model_performance(model_words):",
  capture.output(diag_perf)
), file.path(script_dir, "diagnostics_report.txt"))
