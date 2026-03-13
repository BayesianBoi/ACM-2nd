# TODO
# - implement more sophisticated adversary agent

# 0. IMPORTING DEPENDENCIES

set.seed(123)
pacman::p_load(cmdstanr, posterior, tidyverse, cowplot, bayesplot)


# 1. LOAD MODEL

model <- cmdstan_model("./RL.stan",
                       cpp_options = list(stan_threads = FALSE))
T <- 100
rate_opponent <- 0.75  # biased opponent

# 2. SIMULATION FUNCTION

simulate_rl_mp <- function(alpha, tau, T, rate_opponent) {
  V <- numeric(T)
  choice <- integer(T)
  opponent_choice <- integer(T)

  V[1] <- 0.5

  for (t in 1:T) {

    p <- plogis(tau * (2 * V[t] - 1))
    choice[t] <- rbinom(1, 1, p)

    opponent_choice[t] <- rbinom(1, 1, rate_opponent)

    if (t < T) {
      V[t+1] <- V[t] + alpha * (opponent_choice[t] - V[t])
    }
  }
  
  list(choice = choice, opponent_choice = opponent_choice)
}

compute_ebfmi <- function(energy_vec) {
  if (length(energy_vec) < 2 || isTRUE(all.equal(stats::var(energy_vec), 0))) {
    return(NA_real_)
  }
  mean(diff(energy_vec)^2) / stats::var(energy_vec)
}

# 3. PRIOR PREDICTIVE CHECK

dummy_data <- list(
  T = T,
  choice = rep(0, T),
  opponent_choice = rep(0, T),
  alpha_prior_params = 2,
  tau_prior_sd = 1.5
)

fit_prior <- model$sample(
  data = dummy_data,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_sampling = 500,
  refresh = 500,
  iter_warmup = 0,
  fixed_param = TRUE
  #adapt_delta = 0.75 
)

# Use draw-level expected rates from predictive probabilities.
# This summarizes each posterior/prior draw by its mean predicted rate over trials
# (continuous, bounded in [0,1], and less noisy than averaging binary choices).
draws_prior_prob <- as_draws_matrix(fit_prior$draws("choice_prob_priorp"))
prior_prob_cols <- grep("^choice_prob_priorp\\[", colnames(draws_prior_prob))
prior_rates <- rowMeans(draws_prior_prob[, prior_prob_cols])

# 4. POSTERIOR PREDICTIVE CHECK

sim_data <- simulate_rl_mp(
  alpha = 0.6,
  tau = 3,
  T = T,
  rate_opponent = rate_opponent
)
data_list <- list(
  T = T,
  choice = sim_data$choice,
  opponent_choice = sim_data$opponent_choice,
  alpha_prior_params = 2,
  tau_prior_sd = 1.5
)
fit_post <- model$sample(
  data = data_list,
  seed = 123,
  chains = 4,
  iter_sampling = 1000,
  iter_warmup = 1000
)

# 4a. MCMC DIAGNOSTICS (HAIRY CATERPILLAR + TABLES)

trace_vars <- c("alpha", "tau")
trace_draws <- fit_post$draws(variables = trace_vars, format = "draws_array")

p_mcmc_trace <- bayesplot::mcmc_trace(trace_draws) +
  facet_wrap(~parameter, ncol = 1, scales = "free_y") +
  labs(title = "MCMC Trace Plot") +
  theme_cowplot()

param_diag_tbl <- fit_post$summary(
  variables = c("alpha", "tau")
) %>%
  select(any_of(c(
    "variable", "mean", "sd", "rhat", "ess_bulk", "ess_tail", "mcse_mean", "mcse_sd"
  )))

sampler_diag_draws <- posterior::as_draws_df(
  fit_post$sampler_diagnostics(format = "draws_df")
)

ebfmi_by_chain <- sampler_diag_draws %>%
  group_by(.chain) %>%
  summarise(ebfmi = compute_ebfmi(energy__), .groups = "drop")

sampler_diag_tbl <- tibble(
  metric = c(
    "Total divergences",
    "Transitions at max treedepth (10)",
    "Min E-BFMI across chains",
    "Max R-hat (key params)",
    "Min bulk ESS (key params)",
    "Min tail ESS (key params)"
  ),
  value = c(
    sum(sampler_diag_draws$divergent__),
    sum(sampler_diag_draws$treedepth__ >= 10),
    min(ebfmi_by_chain$ebfmi, na.rm = TRUE),
    max(param_diag_tbl$rhat, na.rm = TRUE),
    min(param_diag_tbl$ess_bulk, na.rm = TRUE),
    min(param_diag_tbl$ess_tail, na.rm = TRUE)
  ),
  guideline = c(
    "0",
    "0",
    "> 0.3",
    "< 1.01",
    "as large as possible",
    "as large as possible"
  )
)

print(param_diag_tbl)
print(sampler_diag_tbl)

draws_post_prob <- as_draws_matrix(fit_post$draws("choice_prob_postp"))
post_prob_cols <- grep("^choice_prob_postp\\[", colnames(draws_post_prob))
post_rates <- rowMeans(draws_post_prob[, post_prob_cols])

prior_post_df <- bind_rows(
  tibble(rate = prior_rates, source = "Prior predictive"),
  tibble(rate = post_rates, source = "Posterior predictive")
)

# Most interpretable PPC on a 0-1 rate scale:
# each point is one draw's overall predicted rate.
p_prior_post <- ggplot(prior_post_df, aes(x = rate, fill = source, colour = source)) +
  geom_density(alpha = 0.25, linewidth = 0.8) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey50") +
  geom_vline(xintercept = rate_opponent, colour = "#2C7FB8", linewidth = 0.9) +
  geom_vline(xintercept = mean(sim_data$choice), colour = "#E84855", linewidth = 0.8) +
  annotate(
    "text", x = rate_opponent, y = Inf,
    label = paste0("Opponent rate = ", round(rate_opponent, 2)),
    hjust = -0.1, vjust = 1.8, size = 3.5, colour = "#2C7FB8"
  ) +
  annotate(
    "text", x = mean(sim_data$choice), y = Inf,
    label = paste0("Observed rate = ", round(mean(sim_data$choice), 2)),
    hjust = -0.1, vjust = 3.2, size = 3.5, colour = "#E84855"
  ) +
  labs(
    title = "Prior vs Posterior Predictive Mean Choice Rate",
    x = "Rate (0-1)",
    y = "Density",
    fill = "",
    colour = ""
  ) +
  coord_cartesian(xlim = c(0, 1)) +
  scale_x_continuous(breaks = seq(0, 1, 0.1)) +
  theme_cowplot()

# Additional "main-style" plot on Mean Prob(choice = 1), using binary predictive choices.
draws_prior_bin <- as_draws_matrix(fit_prior$draws("choice_priorp"))
prior_bin_cols <- grep("^choice_priorp\\[", colnames(draws_prior_bin))
prior_mean_prob <- rowMeans(draws_prior_bin[, prior_bin_cols])

draws_post_bin <- as_draws_matrix(fit_post$draws("choice_postp"))
post_bin_cols <- grep("^choice_postp\\[", colnames(draws_post_bin))
post_mean_prob <- rowMeans(draws_post_bin[, post_bin_cols])

prior_post_meanprob_df <- bind_rows(
  tibble(mean_prob = prior_mean_prob, source = "Prior predictive"),
  tibble(mean_prob = post_mean_prob, source = "Posterior predictive")
)

p_prior_post_meanprob <- ggplot(
  prior_post_meanprob_df,
  aes(x = mean_prob, fill = source, colour = source)
) +
  geom_density(alpha = 0.25, linewidth = 0.8) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey50") +
  geom_vline(xintercept = mean(sim_data$choice), colour = "#E84855", linewidth = 0.8) +
  annotate(
    "text", x = mean(sim_data$choice), y = Inf,
    label = paste0("Observed mean = ", round(mean(sim_data$choice), 2)),
    hjust = -0.1, vjust = 1.8, size = 3.5, colour = "#E84855"
  ) +
  labs(
    title = "Prior vs Posterior Predictive Mean Prob(choice = 1)",
    x = "Mean Prob(choice = 1)",
    y = "Density",
    fill = "",
    colour = ""
  ) +
  coord_cartesian(xlim = c(0, 1)) +
  theme_cowplot()

# 5. SIMULATION-BASED CALIBRATION (SBC)

n_sbc <- 20
sbc_results <- vector("list", n_sbc)

for (i in 1:n_sbc) {
  alpha_true <- rbeta(1, 2, 2)
  tau_true <- rlnorm(1, meanlog = 0, sdlog = 1.5)

  sim_sbc <- simulate_rl_mp(
    alpha = alpha_true,
    tau = tau_true,
    T = T,
    rate_opponent = rate_opponent
  )

  sbc_data <- list(
    T = T,
    choice = sim_sbc$choice,
    opponent_choice = sim_sbc$opponent_choice,
    alpha_prior_params = 2,
    tau_prior_sd = 1.5
  )

  fit_sbc <- model$sample(
    data = sbc_data,
    seed = 1000 + i,
    chains = 2,
    parallel_chains = 2,
    iter_sampling = 500,
    iter_warmup = 500,
    refresh = 0
  )

  sbc_draws <- as_draws_df(fit_sbc$draws(c("alpha", "tau")))
  n_draws <- nrow(sbc_draws)

  sbc_results[[i]] <- tibble(
    rep = i,
    parameter = c("alpha", "tau"),
    rank = c(
      sum(sbc_draws$alpha < alpha_true),
      sum(sbc_draws$tau < tau_true)
    ),
    n_draws = n_draws
  )
}

sbc_df <- bind_rows(sbc_results) %>%
  mutate(rank_scaled = rank / n_draws)

p_sbc <- ggplot(sbc_df, aes(x = rank_scaled)) +
  geom_histogram(bins = 10, fill = "royalblue3", colour = "black", alpha = 0.6) +
  geom_hline(yintercept = n_sbc / 10, linetype = "dashed", colour = "#E84855") +
  facet_wrap(~parameter, scales = "free_y") +
  labs(
    title = "SBC Rank Histograms",
    subtitle = "Dashed line is expected count per bin under calibration",
    x = "Scaled rank",
    y = "Count"
  ) +
  theme_cowplot()

sbc_diag_tbl <- sbc_df %>%
  group_by(parameter) %>%
  summarise(
    mean_rank_scaled = mean(rank_scaled),
    sd_rank_scaled = sd(rank_scaled),
    .groups = "drop"
  )

print(sbc_diag_tbl)

# 6. FULL JOINT PARAMETER RECOVERY

alpha_grid  <- c(0.1, 0.5, 0.9)
tau_grid    <- c(0.1, 5, 10)

n_reps <- 1

recovery_results <- list()
counter <- 1

for (alpha_true in alpha_grid) {
  for (tau_true in tau_grid) {
    for (rep in 1:n_reps) {
      
      sim <- simulate_rl_mp(
        alpha = alpha_true,
        tau = tau_true,
        T = T,
        rate_opponent = rate_opponent
      )
      
      data_list <- list(
        T = T,
        choice = sim$choice,
        opponent_choice = sim$opponent_choice,
        alpha_prior_params = 2,
        tau_prior_sd = 1.5
      )
      
      fit <- model$sample(
        data = data_list,
        seed = 123,
        chains = 4,
        iter_sampling = 1000,
        iter_warmup = 1000,
        refresh = 0
      )
      
      draws <- as_draws_df(fit$draws())
      
      recovery_results[[counter]] <- data.frame(
        alpha_true = alpha_true,
        alpha_est = mean(draws$alpha),
        alpha_sd = sd(draws$alpha),
        tau_true = tau_true,
        tau_sd = sd(draws$tau),
        tau_est = mean(draws$tau)
      )
      
      counter <- counter + 1
    }
  }
}

recovery_df <- bind_rows(recovery_results)

# 7. RECOVERY PLOTS

recovery_long <- recovery_df %>%
  select(alpha_true, tau_true, alpha_est, tau_est) %>%
  pivot_longer(
    cols = c(alpha_est, tau_est),
    names_to = "parameter",
    values_to = "estimate"
  ) %>%
  mutate(
    true_value = case_when(
      parameter == "alpha_est" ~ alpha_true,
      parameter == "tau_est"   ~ tau_true
    ))

## Actual plotting 

p1 <- ggplot(recovery_long, aes(x = true_value, y = estimate, colour = parameter)) +
  geom_point(alpha = 0.8, size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey30") +
  ylab("Estimated Value") +
  xlab("Ground Truth Value") +
  facet_wrap(~parameter, scales = "free") +
  scale_colour_brewer(palette = "Set2", guide = "none") +
  theme_cowplot() +
  labs(title = "Joint Parameter Recovery")

# Posterior SDs
p2 <- ggplot(recovery_df, aes(x = factor(alpha_true), y = alpha_sd, fill = factor(alpha_true))) +
  geom_boxplot(alpha = 0.5, linewidth = 0.4, width = 0.5) +
  geom_jitter(aes(colour = factor(alpha_true)), width = 0.15, alpha = 0.9, size = 1.2) +
  labs(
    title = "Posterior SD of Alpha",
    x = "True Alpha",
    y = "Posterior SD"
  ) +
  theme_cowplot() + 
  theme(legend.position = "none")

p3 <- ggplot(recovery_df, aes(x = factor(tau_true), y = tau_sd, fill = factor(tau_true))) +
  geom_boxplot(alpha = 0.5, linewidth = 0.4, width = 0.5) + 
  geom_jitter(aes(colour = factor(tau_true)), width = 0.15, alpha = 0.9, size = 1.2) +
  labs(
    title = "Posterior SD of Tau",
    x = "True Tau",
    y = "Posterior SD"
  ) +
  theme_cowplot() +
  theme(legend.position = "none")

## ud i æteren 

p_prior_post
p_prior_post_meanprob
p_mcmc_trace
param_diag_tbl
sampler_diag_tbl
p_sbc
sbc_diag_tbl
p1
p2
p3
