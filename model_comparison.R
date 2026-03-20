# MODEL COMPARISON: RL Agent vs. Biased Agent
#
# Fits both models on both datasets and compares via PSIS-LOO.
# Fitted objects are cached as .rds files to avoid refitting.
# Outputs: model_comparison_plots.pdf

set.seed(123)
pacman::p_load(cmdstanr, posterior, tidyverse, cowplot, loo, bayesplot)

# ── 0. LOAD MODELS ────────────────────────────────────────────────────────────

model_rl     <- cmdstan_model("./RL.stan",     cpp_options = list(stan_threads = FALSE))
model_biased <- cmdstan_model("./biased.stan", cpp_options = list(stan_threads = FALSE))

n_trials     <- 100
block_size   <- 20
block_biases <- c(0.1, 0.2, 0.3, 0.4, 0.5)

# ── 1. SIMULATION FUNCTIONS ───────────────────────────────────────────────────

simulate_rl_agent <- function(alpha, tau, n_trials, block_biases, block_size) {
  prob  <- numeric(n_trials)
  choice <- integer(n_trials)
  opp    <- integer(n_trials)
  prob[1] <- 0.5

  for (t in seq_len(n_trials)) {
    choice[t] <- rbinom(1, 1, plogis(tau * qlogis(prob[t])))
    opp[t]    <- rbinom(1, 1, block_biases[ceiling(t / block_size)])
    if (t < n_trials)
      prob[t + 1] <- prob[t] + alpha * (opp[t] - prob[t])
  }
  list(choice = choice, opponent_choice = opp)
}

simulate_biased_agent <- function(theta, n_trials) {
  list(choice = rbinom(n_trials, 1, theta))
}

# ── 2. SIMULATE DATA ──────────────────────────────────────────────────────────

rl_sim     <- simulate_rl_agent(alpha = 0.5, tau = 5,
                                n_trials = n_trials,
                                block_biases = block_biases,
                                block_size = block_size)

biased_sim <- simulate_biased_agent(theta = 0.7, n_trials = n_trials)

# Neutral opponent sequence for the RL model when applied to biased-agent data
opponent_seq <- rbinom(n_trials, 1, 0.5)

# ── 3. DATA LISTS ─────────────────────────────────────────────────────────────

rl_data_rl <- list(
  n_trials = n_trials, choice = rl_sim$choice,
  opponent_choice = rl_sim$opponent_choice,
  alpha_prior_shapes = 2, tau_prior_sd = 1.5, initial_prob_choice = 0.5
)

rl_data_biased <- list(
  n_trials = n_trials, choice = biased_sim$choice,
  opponent_choice = opponent_seq,
  alpha_prior_shapes = 2, tau_prior_sd = 1.5, initial_prob_choice = 0.5
)

biased_data_biased <- list(
  n_trials = n_trials, choice = biased_sim$choice,
  theta_prior_shape1 = 1, theta_prior_shape2 = 1
)

biased_data_rl <- list(
  n_trials = n_trials, choice = rl_sim$choice,
  theta_prior_shape1 = 1, theta_prior_shape2 = 1
)

# ── 4. FIT (with .rds caching) ────────────────────────────────────────────────

sample_or_load <- function(rds_path, model, data, ...) {
  if (file.exists(rds_path)) {
    message("Loading cached fit: ", rds_path)
    readRDS(rds_path)
  } else {
    fit <- model$sample(data = data, seed = 123, chains = 4,
                        iter_warmup = 1000, iter_sampling = 1000,
                        refresh = 0, ...)
    saveRDS(fit, rds_path)
    fit
  }
}

fit_rl_on_rl         <- sample_or_load("fit_rl_on_rl.rds",         model_rl,     rl_data_rl)
fit_biased_on_rl     <- sample_or_load("fit_biased_on_rl.rds",     model_biased, biased_data_rl)
fit_rl_on_biased     <- sample_or_load("fit_rl_on_biased.rds",     model_rl,     rl_data_biased)
fit_biased_on_biased <- sample_or_load("fit_biased_on_biased.rds", model_biased, biased_data_biased)

# ── 5. DIAGNOSTICS ────────────────────────────────────────────────────────────

for (nm in c("fit_rl_on_rl", "fit_biased_on_rl", "fit_rl_on_biased", "fit_biased_on_biased")) {
  cat("\n──", nm, "──\n")
  get(nm)$diagnostic_summary()
}

# ── 6. PSIS-LOO ───────────────────────────────────────────────────────────────

loo_rl_on_rl         <- fit_rl_on_rl$loo(save_psis = TRUE)
loo_biased_on_rl     <- fit_biased_on_rl$loo(save_psis = TRUE)
loo_rl_on_biased     <- fit_rl_on_biased$loo(save_psis = TRUE)
loo_biased_on_biased <- fit_biased_on_biased$loo(save_psis = TRUE)

cat("\n=== RL-generated data ===\n")
print(loo_compare(loo_rl_on_rl, loo_biased_on_rl))

cat("\n=== Biased-generated data ===\n")
print(loo_compare(loo_rl_on_biased, loo_biased_on_biased))

# ── 7. PLOTS ──────────────────────────────────────────────────────────────────

theme_set(theme_classic())

strip_theme <- theme(
  strip.background = element_rect(fill = "gray95", color = NA),
  strip.text       = element_text(size = 10)
)

# Helper: ELPD bar chart for one dataset
elpd_plot <- function(loo1, loo2, label1, label2, data_label) {
  df <- data.frame(
    model = c(label1, label2),
    elpd  = c(loo1$estimates["elpd_loo", "Estimate"],
              loo2$estimates["elpd_loo", "Estimate"]),
    se    = c(loo1$estimates["elpd_loo", "SE"],
              loo2$estimates["elpd_loo", "SE"])
  )
  ggplot(df, aes(x = model, y = elpd, colour = model)) +
    geom_point(size = 4) +
    geom_errorbar(aes(ymin = elpd - se, ymax = elpd + se),
                  width = 0.15, linewidth = 0.8) +
    scale_colour_manual(values = c("royalblue3", "#E84855"), guide = "none") +
    labs(title = data_label, x = NULL, y = "ELPD-LOO (higher = better)") +
    theme_classic() + strip_theme
}

# Helper: pointwise LOO difference plot
pointwise_plot <- function(loo1, loo2, label1, label2, data_label) {
  diff <- loo1$pointwise[, "elpd_loo"] - loo2$pointwise[, "elpd_loo"]
  df   <- data.frame(trial = seq_along(diff), diff = diff)
  ggplot(df, aes(x = trial, y = diff)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "darkred") +
    geom_point(alpha = 0.5, size = 1.5, colour = "gray40") +
    geom_smooth(method = "loess", se = FALSE, colour = "royalblue3", linewidth = 0.8) +
    labs(
      title    = data_label,
      subtitle = paste0("Pointwise ELPD: ", label1, " − ", label2),
      x        = "Trial", y = "ELPD difference"
    ) +
    theme_classic() + strip_theme
}

# Helper: Pareto k diagnostic plot
pareto_plot <- function(loo_obj, title) {
  k   <- loo_obj$diagnostics$pareto_k
  df  <- data.frame(obs = seq_along(k), k = k,
                    flag = ifelse(k > 0.7, "k > 0.7", "ok"))
  ggplot(df, aes(x = obs, y = k, colour = flag)) +
    geom_point(size = 1.5, alpha = 0.7) +
    geom_hline(yintercept = c(0.5, 0.7), linetype = "dashed", color = "darkred") +
    annotate("text", x = max(df$obs), y = 0.72, label = "k = 0.7",
             hjust = 1, size = 3, colour = "darkred") +
    annotate("text", x = max(df$obs), y = 0.52, label = "k = 0.5",
             hjust = 1, size = 3, colour = "darkred") +
    scale_colour_manual(values = c("k > 0.7" = "#E84855", "ok" = "gray50"),
                        guide = "none") +
    labs(title = title, x = "Observation", y = "Pareto k") +
    theme_classic() + strip_theme
}

# Build all panels
p_elpd_rl     <- elpd_plot(loo_rl_on_rl, loo_biased_on_rl,
                            "RL model", "Biased model", "RL-generated data")
p_elpd_biased <- elpd_plot(loo_rl_on_biased, loo_biased_on_biased,
                            "RL model", "Biased model", "Biased-generated data")

p_pw_rl     <- pointwise_plot(loo_rl_on_rl, loo_biased_on_rl,
                               "RL", "Biased", "RL-generated data")
p_pw_biased <- pointwise_plot(loo_rl_on_biased, loo_biased_on_biased,
                               "RL", "Biased", "Biased-generated data")

p_k_rl_on_rl         <- pareto_plot(loo_rl_on_rl,         "RL model | RL data")
p_k_biased_on_rl     <- pareto_plot(loo_biased_on_rl,     "Biased model | RL data")
p_k_rl_on_biased     <- pareto_plot(loo_rl_on_biased,     "RL model | Biased data")
p_k_biased_on_biased <- pareto_plot(loo_biased_on_biased, "Biased model | Biased data")

# ── 8. SAVE TO PDF ────────────────────────────────────────────────────────────

pdf("model_comparison_plots.pdf", width = 10, height = 7)

# Page 1: ELPD comparison
print(plot_grid(p_elpd_rl, p_elpd_biased, ncol = 2, labels = c("A", "B"),
                label_size = 12))

# Page 2: Pointwise ELPD differences
print(plot_grid(p_pw_rl, p_pw_biased, ncol = 2, labels = c("C", "D"),
                label_size = 12))

# Page 3: Pareto k diagnostics
print(plot_grid(p_k_rl_on_rl, p_k_biased_on_rl,
                p_k_rl_on_biased, p_k_biased_on_biased,
                ncol = 2, labels = c("E", "F", "G", "H"),
                label_size = 12))

dev.off()
message("Saved: model_comparison_plots.pdf")
