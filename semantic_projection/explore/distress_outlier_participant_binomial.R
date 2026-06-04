script_args <- commandArgs(trailingOnly = TRUE)

default_input <- "semantic_projection_qwen.csv"
input_file <- if (length(script_args) >= 1) script_args[[1]] else default_input

base_name <- tools::file_path_sans_ext(basename(input_file))
suffix <- sub("^semantic_projection_", "", base_name)

project_dir <- normalizePath(getwd())
if (basename(project_dir) == "explore") {
  project_dir <- dirname(dirname(project_dir))
} else if (basename(project_dir) == "semantic_projection") {
  project_dir <- dirname(project_dir)
}

explore_dir <- file.path(project_dir, "semantic_projection", "explore")
input_path <- if (grepl("^/", input_file)) input_file else file.path(explore_dir, input_file)

library(dplyr)
library(emmeans)

if (!file.exists(input_path)) {
  stop(sprintf("Input file not found: %s", input_path))
}

data <- read.csv(input_path, stringsAsFactors = FALSE)

required_cols <- c("Participant", "condition", "projection_fear_vs_calm")
missing_cols <- setdiff(required_cols, names(data))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
}

data$condition <- factor(as.character(data$condition), levels = c("3", "1", "2"))

mad0 <- median(abs(data$projection_fear_vs_calm), na.rm = TRUE)
multiplier <- 1.00
upper_fence <- multiplier * mad0
lower_fence <- -multiplier * mad0

data$is_distress_outlier <- data$projection_fear_vs_calm > upper_fence
data$is_relaxed_outlier <- data$projection_fear_vs_calm < lower_fence

participant_df <- data %>%
  group_by(Participant, condition) %>%
  summarise(
    n_words = n(),
    distress_outliers = sum(is_distress_outlier, na.rm = TRUE),
    non_distress_words = n_words - distress_outliers,
    distress_prop = distress_outliers / n_words,
    .groups = "drop"
  ) %>%
  arrange(condition, Participant)

binom_model <- glm(
  cbind(distress_outliers, non_distress_words) ~ condition,
  data = participant_df,
  family = quasibinomial()
)

binom_summary <- summary(binom_model)
binom_anova <- anova(binom_model, test = "F")

emm_prob <- emmeans(binom_model, ~ condition, type = "response")
emm_prob_pairs <- pairs(emm_prob)

prop_lm <- lm(distress_prop ~ condition, data = participant_df)
prop_summary <- summary(prop_lm)
prop_emm <- emmeans(prop_lm, ~ condition)
prop_pairs <- pairs(prop_emm)

participant_out <- file.path(explore_dir, sprintf("distress_outlier_participant_counts_%s.csv", suffix))
report_out <- file.path(explore_dir, sprintf("distress_outlier_participant_report_%s.txt", suffix))

write.csv(participant_df, participant_out, row.names = FALSE)

report_lines <- c(
  sprintf("Participant-level distress-extreme report: %s", suffix),
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  sprintf("Input file: %s", input_path),
  "Zero-centered thresholding rule:",
  sprintf("MAD0 = %.6f", mad0),
  sprintf("Multiplier = %.2f", multiplier),
  sprintf("Lower threshold = %.6f", lower_fence),
  sprintf("Upper threshold = %.6f", upper_fence),
  "",
  "Participant-level counts:",
  capture.output(print(participant_df, row.names = FALSE)),
  "",
  "Binomial model: cbind(distress_outliers, non_distress_words) ~ condition",
  capture.output(print(binom_summary)),
  "",
  "Binomial model omnibus test:",
  capture.output(print(binom_anova)),
  "",
  "Estimated marginal distress-outlier probabilities by condition:",
  capture.output(print(emm_prob)),
  "",
  "Pairwise comparisons of distress-outlier probabilities:",
  capture.output(print(emm_prob_pairs)),
  "",
  "Proportion LM sensitivity analysis: distress_prop ~ condition",
  capture.output(print(prop_summary)),
  "",
  "Estimated marginal means for distress proportion:",
  capture.output(print(prop_emm)),
  "",
  "Pairwise comparisons of distress proportion:",
  capture.output(print(prop_pairs))
)

writeLines(report_lines, report_out)
cat(paste(report_lines, collapse = "\n"), "\n")
