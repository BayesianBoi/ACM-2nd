# 0. IMPORTING DEPENDENCIES

set.seed(123)
pacman::p_load(cmdstanr, posterior, tidyverse, cowplot, priorsense)

# 1. LOAD MODEL

model <- cmdstan_model("./RL.stan", cpp_options = list(stan_threads = FALSE))
n_trials <- 100

# volatile opponent: bias changes every block
block_size <- 20
block_biases <- c(0.1, 0.2, 0.3, 0.4, 0.5)

# 2. SIMULATION FUNCTION

simulate_rl_mp <- function(alpha, tau, n_trials, block_biases, block_size) {
  expected_prob <- numeric(n_trials)
  choice <- integer(n_trials)
  opponent_choice <- integer(n_trials)

  expected_prob[1] <- 0.5

  for (t in 1:n_trials) {
    choice[t] <- rbinom(1, 1, plogis(tau * qlogis(expected_prob[t])))

    block <- ceiling(t / block_size)
    opponent_choice[t] <- rbinom(1, 1, block_biases[block])

    if (t < n_trials) {
      expected_prob[t + 1] <- expected_prob[t] + alpha * (opponent_choice[t] - expected_prob[t])
    }
  }

  list(
    choice = choice,
    expected_prob = expected_prob,
    opponent_choice = opponent_choice,
    block_biases = block_biases,
    alpha = alpha,
    tau = tau
  )
}

sim_data <- simulate_rl_mp(
  alpha = 0.6,
  tau = 5,
  n_trials = n_trials,
  block_biases = block_biases,
  block_size = block_size
)

data_list <- list(
  n_trials = n_trials,
  choice = sim_data$choice,
  opponent_choice = sim_data$opponent_choice,
  alpha_prior_shapes = 2,
  tau_prior_sd = 2.5,
  initial_prob_choice = 0.5
)

model_fit <- model$sample(
  data = data_list,
  seed = 123,
  chains = 4,
  iter_sampling = 1000,
  iter_warmup = 1000
)

# 3. PRIOR PREDICTIVE CHECK

draws_prior <- as_draws_matrix(model_fit$draws("choice_priorp"))
choice_cols <- grep("choice_priorp", colnames(draws_prior))
prior_rep <- draws_prior[, choice_cols]

# gammel version: colMeans averages over draws per trial. Since the prior predictive does
# not update expected_prob (stays at 0.5), all trials are exchangeable, so
# colMeans gives ~100 near-identical values — a meaningless spike.
# p_prior <- ggplot(data.frame(mean_prob = colMeans(prior_rep)), aes(x = mean_prob)) +

# rowMeans gives the overall choice rate per draw, showing the distribution of
# agent behaviours the prior produces
p_prior <- ggplot(data.frame(mean_prob = colMeans(prior_rep)), aes(x = mean_prob)) +
  geom_histogram(
    fill = "royalblue3",
    colour = "black",
    alpha = 0.4,
    bounds = c(0, 1)
  ) +
  geom_vline(
    xintercept = 0.5,
    linetype = "dashed",
    colour = "#E84855"
  ) +
  annotate(
    "text",
    x = 0.5,
    y = Inf,
    label = "Chance (0.5)",
    hjust = -0.1,
    vjust = 1.5,
    size = 3.5,
    colour = "#E84855"
  ) +
  coord_cartesian(
    xlim = c(0, 1),
    ylim = c(0, NA),
    expand = c(0, 0)
  ) +
  labs(title = "Prior Predictive Overall Choice Rate", x = "Prob(choice = 1)", y = "Density") +
  theme_cowplot()

p_prior

true_bias_per_trial <- rep(block_biases, each = block_size)

# learning trajectory: agent's expected_prob vs true block biases
prior_trajectory_df <- data.frame(
  trial = 1:n_trials,
  agent_mean = colMeans(prior_rep),
  agent_lower = apply(prior_rep, 2, quantile, 0.05),
  agent_upper = apply(prior_rep, 2, quantile, 0.95),
  true_bias = true_bias_per_trial,
  opponent = sim_data$opponent_choice # plotting the opponent choice also
)

p_prior_trajectory <- ggplot(prior_trajectory_df, aes(x = trial)) +
  geom_point(aes(y = opponent, colour = "Opponent choices"), size = 1.2, alpha = 0.4) +
  geom_step(aes(y = true_bias, colour = "True bias"), linewidth = 1, linetype = "dashed") +
  # geom_ribbon(aes(ymin = agent_lower, ymax = agent_upper), fill = "royalblue3", alpha = 0.25) +
  geom_line(aes(y = agent_mean, colour = "Agent belief (prior)"), linewidth = 0.8) +
  scale_colour_manual(
    values = c("Opponent choices" = "grey50",
               "True bias" = "#E84855",
               "Agent belief (prior)" = "royalblue3")
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    title = "Prior Predictive: Belief Trajectory",
    x = "Trial",
    y = "P(penny is in right hand)",
    colour = NULL
  ) +
  theme_cowplot() +
  theme(legend.position = "bottom")

p_prior_trajectory

# 4. POSTERIOR PREDICTIVE CHECK

draws_post <- as_draws_matrix(model_fit$draws("choice_prob_postp"))
choice_cols <- grep("choice_prob_postp", colnames(draws_post))
post_rep <- draws_post[, choice_cols]

p_posterior <- ggplot(data.frame(mean_prob = colMeans(post_rep)), aes(x = mean_prob)) +
  geom_histogram(
    fill = "royalblue3",
    colour = "black",
    alpha = 0.4,
    bounds = c(0, 1)
  ) +
  geom_vline(
    xintercept = 0.5,
    linetype = "dashed",
    colour = "#E84855"
  ) +
  annotate(
    "text",
    x = 0.5,
    y = Inf,
    label = "Chance (0.5)",
    hjust = -0.1,
    vjust = 1.5,
    size = 3.5,
    colour = "#E84855"
  ) +
  coord_cartesian(
    xlim = c(0, 1),
    ylim = c(0, NA),
    expand = c(0, 0)
  ) +
  labs(title = "Posterior Predictive Overall Choice Rate", x = "Prob(choice = 1)", y = "Density") +
  theme_cowplot()

p_posterior

# build the true block bias per trial for the step function
true_bias_per_trial <- rep(block_biases, each = block_size)

# learning trajectory: agent's expected_prob vs true block biases
post_trajectory_df <- data.frame(
  trial = 1:n_trials,
  agent_mean = colMeans(post_rep),
  agent_lower = apply(post_rep, 2, quantile, 0.05),
  agent_upper = apply(post_rep, 2, quantile, 0.95),
  true_bias = true_bias_per_trial,
  opponent = sim_data$opponent_choice # plotting the opponent choice also
)

p_posterior_trajectory <- ggplot(post_trajectory_df, aes(x = trial)) +
  geom_point(aes(y = opponent, colour = "Opponent choices"), size = 1.2, alpha = 0.4) +
  geom_step(aes(y = true_bias, colour = "True bias"), linewidth = 1, linetype = "dashed") +
  # geom_ribbon(aes(ymin = agent_lower, ymax = agent_upper), fill = "royalblue3", alpha = 0.25) +
  geom_line(aes(y = agent_mean, colour = "Agent belief (posterior)"), linewidth = 0.8) +
  scale_colour_manual(
    values = c("Opponent choices" = "grey50",
               "True bias" = "#E84855",
               "Agent belief (posterior)" = "royalblue3")
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    title = "Posterior Predictive: Belief Trajectory",
    x = "Trial",
    y = "P(penny is in right hand)",
    colour = NULL
  ) +
  theme_cowplot() +
  theme(legend.position = "bottom")

p_posterior_trajectory



## Prior-Posterior Updates 

draws_alpha_prior <- as_draws_matrix(model_fit$draws("alpha_prior"))[, 1]
draws_tau_prior   <- as_draws_matrix(model_fit$draws("tau_prior"))[, 1]

draws_alpha_post  <- as_draws_matrix(model_fit$draws("alpha"))[, 1]
draws_tau_post    <- as_draws_matrix(model_fit$draws("tau"))[, 1]

draws_alpha_prior_df <- data.frame(alpha = draws_alpha_prior)
draws_alpha_post_df  <- data.frame(alpha = draws_alpha_post)
draws_tau_prior_df   <- data.frame(tau = draws_tau_prior)
draws_tau_post_df    <- data.frame(tau = draws_tau_post)

## Alpha

plot_alpha <- ggplot() +
  geom_density(
    data   = draws_alpha_prior_df, aes(x = alpha_prior),
    fill   = "grey60",
    colour = "grey40",
    alpha  = 0.35,
    bounds = c(0, 1)
  ) +
  geom_density(
    data   = draws_alpha_post_df, aes(x = alpha),
    fill   = "royalblue3",
    colour = "black",
    alpha  = 0.40,
    bounds = c(0, 1)
  ) +
  annotate("text", x = 0.8, y = Inf, label = "Prior",     hjust = 0, vjust = 1.5, size = 5, colour = "grey40") +
  annotate("text", x = 0.8, y = Inf, label = "Posterior", hjust = 0, vjust = 3.2, size = 5, colour = "royalblue3") +
    geom_vline(xintercept = sim_data$alpha, linetype = "dashed", colour = "#E84855", linewidth = 0.8) +
    annotate("text", x = sim_data$alpha, y = Inf, label = paste0("True (", sim_data$alpha, ")"),
             hjust = -0.1, vjust = 1.5, size = 3.5, colour = "#E84855") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, NA), expand = c(0, 0)) +
  labs(
    title = "Prior–Posterior update: alpha",
    x     = "alpha",
    y     = "Density"
  ) +
  theme_cowplot()

## Tau

plot_tau <- ggplot() +
  geom_density(
    data   = draws_tau_prior_df, aes(x = tau_prior),
    fill   = "grey60",
    colour = "grey40",
    alpha  = 0.35,
    n=100000
  ) +
  geom_density(
    data   = draws_tau_post_df, aes(x = tau),
    fill   = "royalblue3",
    colour = "black",
    alpha  = 0.40,
    n=100000
  ) +
  annotate("text", x = 12, y = Inf, label = "Prior",     hjust = 0, vjust = 1.5, size = 5, colour = "grey40") +
  annotate("text", x = 12, y = Inf, label = "Posterior", hjust = 0, vjust = 3.2, size = 5, colour = "royalblue3") +
  geom_vline(xintercept = sim_data$tau, linetype = "dashed", colour = "#E84855", linewidth = 0.8) +
  annotate("text", x = sim_data$tau, y = Inf, label = paste0("True (", sim_data$tau, ")"),
           hjust = -0.1, vjust = 1.5, size = 3.5, colour = "#E84855") +
  coord_cartesian(xlim = c(0, 15), ylim = c(0, NA), expand = c(0, 0)) +
  labs(
    title = "Prior–Posterior update: tau",
    x     = "tau",
    y     = "Density"
  ) +
  theme_cowplot()

plot_alpha
plot_tau


# 5. FULL JOINT PARAMETER RECOVERY

alpha_grid <- c(0.1, 0.5, 0.9)
tau_grid <- c(0.1, 5, 10)
alpha_grid <- c(0.1, 0.5, 0.8)

n_reps <- 3

recovery_results <- list()
counter <- 1

for (alpha_true in alpha_grid) {
  for (tau_true in tau_grid) {
    for (rep in 1:n_reps) {
      sim <- simulate_rl_mp(
        alpha = alpha_true,
        tau = tau_true,
        n_trials = n_trials,
        block_biases = block_biases,
        block_size = block_size
      )

      data_list <- list(
        n_trials = n_trials,
        choice = sim$choice,
        opponent_choice = sim$opponent_choice,
        alpha_prior_shapes = 2,
        tau_prior_sd = 1.5,
        initial_prob_choice = 0.5
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

# 6. RECOVERY PLOTS

recovery_long <- recovery_df %>%
  select(alpha_true, tau_true, alpha_est, tau_est, alpha_sd, tau_sd) %>%
  pivot_longer(
    cols = c(alpha_est, tau_est),
    names_to = "parameter",
    values_to = "estimate"
  ) %>%
  mutate(
    true_value = case_when(
      parameter == "alpha_est" ~ alpha_true,
      parameter == "tau_est" ~ tau_true,
    ),
    sd = case_when(
      parameter == "alpha_est" ~ alpha_sd,
      parameter == "tau_est" ~ tau_sd,
    )
  )

# 7. PRIOR SENSITIVITY ANALYSIS

prior_sense_power_scaling_plot <- powerscale_plot_dens(model_fit, variables=c("alpha", "tau"))

prior_sense_power_scaling_plot

## Actual plotting

parameter_recovery_mean_estimates_plot <- ggplot(
  recovery_long,
  aes(x = true_value, y = estimate, colour = parameter)
) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    colour = "grey30"
  ) +
  # adding error bars now- question is if we should use sd or .05/95 confidence intervals instead
  geom_errorbar(
    aes(ymin = estimate - sd, ymax = estimate + sd),
    width = 0, alpha = 0.5
  ) +
  geom_point(alpha = 0.8, size = 2) +
  ylab("Estimated Value") +
  xlab("Ground Truth Value") +
  facet_wrap(~parameter, scales = "free") +
  scale_colour_brewer(palette = "Set2", guide = "none") +
  theme_cowplot() +
  labs(title = "Joint Parameter Recovery")

parameter_recovery_mean_estimates_plot

# Posterior SDs
parameter_recovery_alpha_plot <- ggplot(recovery_df, aes(
  x = factor(alpha_true),
  y = alpha_sd,
  fill = factor(alpha_true)
)) +
  geom_boxplot(
    alpha = 0.5,
    linewidth = 0.4,
    width = 0.5
  ) +
  geom_jitter(
    aes(colour = factor(alpha_true)),
    width = 0.15,
    alpha = 0.9,
    size = 1.2
  ) +
  labs(title = "Posterior SD of Alpha", x = "True Alpha", y = "Posterior SD") +
  theme_cowplot() +
  theme(legend.position = "none")

parameter_recovery_alpha_plot

parameter_recovery_tau_plot <- ggplot(recovery_df, aes(
  x = factor(tau_true),
  y = tau_sd,
  fill = factor(tau_true)
)) +
  geom_boxplot(
    alpha = 0.5,
    linewidth = 0.4,
    width = 0.5
  ) +
  geom_jitter(
    aes(colour = factor(tau_true)),
    width = 0.15,
    alpha = 0.9,
    size = 1.2
  ) +
  labs(title = "Posterior SD of Tau", x = "True Tau", y = "Posterior SD") +
  theme_cowplot() +
  theme(legend.position = "none")

parameter_recovery_tau_plot
