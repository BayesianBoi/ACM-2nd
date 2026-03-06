# TODO
# - implement more sophisticated adversary agent
# - tidy up imports (tidyverse incorporates most) - done 
# - Harry plotter gets to work on refining visualizations for maximized stakeholder value

# 0. IMPORTING DEPENDENCIES

set.seed(123)
pacman::p_load(cmdstanr, posterior, tidyverse, cowplot)


# 1. LOAD MODEL

model <- cmdstan_model("RL.stan",
                       cpp_options = list(stan_threads = FALSE))
T <- 100
rate_opponent <- 0.75  # biased opponent

# 2. SIMULATION FUNCTION

simulate_rl_mp <- function(alpha, tau, theta, T, rate_opponent) {
  V <- numeric(T)
  choice <- integer(T)
  reward <- integer(T)
  opponent <- integer(T)

  V[1] <- theta

  for (t in 1:T) {
    
    p <- plogis(tau * (2 * V[t] - 1))
    choice[t] <- rbinom(1, 1, p)
    
    opponent[t] <- rbinom(1, 1, rate_opponent)
    
    reward[t] <- as.integer(choice[t] == opponent[t])
    
    if (t < T) {
      V[t+1] <- V[t] + alpha * (reward[t] - V[t])
    }
  }
  
  list(choice = choice, reward = reward)
}

# 3. PRIOR PREDICTIVE CHECK

dummy_data <- list(
  T = T,
  choice = rep(0, T),
  reward = rep(0, T)
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

draws_prior <- as_draws_matrix(fit_prior$draws("choice_priorp"))
choice_cols <- grep("choice_priorp", colnames(draws_prior))
prior_rep <- draws_prior[, choice_cols]

prior_mean_trial <- colMeans(prior_rep)

hist_df <- data.frame(mean_prob = prior_choice_means)

p_prior <- ggplot(hist_df, aes(x = mean_prob)) +
  geom_density(fill = "royalblue3", colour = "black", alpha = 0.4) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey40") +
  annotate("text", x = 0.52, y = Inf, label = "Chance (0.5)",
           hjust = 0, vjust = 1.5, size = 3.5, colour = "#E84855") +
  labs(
    title = "Prior Predictive Overall Choice Rate",
    x = "Mean Prob(choice = 1)",
    y = "Density"
  ) +
  theme_cowplot()


# 4. POSTERIOR PREDICTIVE CHECK

sim_data <- simulate_rl_mp(
  alpha = 0.6,
  tau = 3,
  theta = 0.5,
  T = T,
  rate_opponent = rate_opponent
)
data_list <- list(
  T = T,
  choice = sim_data$choice,
  reward = sim_data$reward
)
fit_post <- model$sample(
  data = data_list,
  seed = 123,
  chains = 4,
  iter_sampling = 1000,
  iter_warmup = 1000
)

draws_post <- as_draws_matrix(fit_post$draws("choice_postp"))
choice_cols <- grep("choice_postp", colnames(draws_post))
post_rep <- draws_post[, choice_cols]

# Overall choice rate distribution
post_choice_means <- rowMeans(post_rep)

hist_df <- data.frame(mean_prob = post_choice_means)

p_post <- ggplot(hist_df, aes(x = mean_prob)) +
  geom_density(fill = "royalblue3", colour = "black", alpha = 0.4) +
  geom_vline(xintercept = mean(sim_data$choice), colour = "#E84855", linewidth = 0.8) +
  annotate("text", x = mean(sim_data$choice), y = Inf,
           label = paste0("true mean = ", round(mean(sim_data$choice), 2)),
           hjust = -0.1, vjust = 1.5, size = 3.5, colour = "#E84855") +
  labs(
    title = "Posterior Predictive Overall Choice Rate",
    x = "Mean Prob(choice = 1)",
    y = "Count"
  ) +
  theme_cowplot()

# 5. FULL JOINT PARAMETER RECOVERY

alpha_grid  <- c(0.1, 0.5, 0.9)
tau_grid    <- c(0.1, 5, 10)
theta_grid  <- c(0.2, 0.5, 0.8)

n_reps <- 1

recovery_results <- list()
counter <- 1

for (alpha_true in alpha_grid) {
  for (tau_true in tau_grid) {
    for (theta_true in theta_grid) {
      for (rep in 1:n_reps) {
        
        sim <- simulate_rl_mp(
          alpha = alpha_true,
          tau = tau_true,
          theta = theta_true,
          T = T,
          rate_opponent = rate_opponent
        )
        
        data_list <- list(
          T = T,
          choice = sim$choice,
          reward = sim$reward
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
          tau_true = tau_true,
          theta_true = theta_true,
          alpha_est = mean(draws$alpha),
          tau_est = mean(draws$tau),
          theta_est = mean(draws$theta),
          alpha_sd = sd(draws$alpha),
          tau_sd = sd(draws$tau),
          theta_sd = sd(draws$theta)
        )
        
        counter <- counter + 1
      }
    }
  }
}

recovery_df <- bind_rows(recovery_results)

# 6. RECOVERY PLOTS

recovery_long <- recovery_df %>%
  select(alpha_true, tau_true, theta_true,
         alpha_est, tau_est, theta_est) %>%
  pivot_longer(
    cols = c(alpha_est, tau_est, theta_est),
    names_to = "parameter",
    values_to = "estimate"
  ) %>%
  mutate(
    true_value = case_when(
      parameter == "alpha_est" ~ alpha_true,
      parameter == "tau_est"   ~ tau_true,
      parameter == "theta_est" ~ theta_true
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

p4 <- ggplot(recovery_df, aes(x = factor(theta_true), y = theta_sd, fill = factor(theta_true))) +
  geom_boxplot(alpha = 0.5, linewidth = 0.4, width = 0.5) +
  geom_jitter(aes(colour = factor(theta_true)), width = 0.15, alpha = 0.9, size = 1.2) +
  labs(
    title = "Posterior SD of Theta",
    x = "True Theta",
    y = "Posterior SD"
  ) +
  theme_cowplot() +
  theme(legend.position = "none")

## ud i æteren 

p_prior
p_post
p1
p2
p3
p4

