# TODO
# - implement more sophisticated adversary agent
# - tidy up imports (tidyverse incorporates most)
# - Harry plotter gets to work on refining visualizations for maximized stakeholder value

# 0. IMPORTING DEPENDENCIES

set.seed(123)
library(cmdstanr)
library(posterior)
library(tidyverse)
library(ggplot2)

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

plot(1:T, prior_mean_trial, type = "l",
     ylim = c(0,1),
     main = "Prior Predictive Mean Choice",
     ylab = "Prob(choice=1)",
     xlab = "Trial")
abline(h = 0.5, lty = 2)

prior_choice_means <- rowMeans(prior_rep)

hist(prior_choice_means, breaks = 30,
     main = "Prior Predictive Overall Choice Rate",
     xlab = "Mean Prob(choice=1)")

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

hist(post_choice_means, breaks = 30,
     main = "Posterior Predictive Overall Choice Rate",
     xlab = "Mean Prob(choice=1)")

abline(v = mean(sim_data$choice), col = "red", lwd = 2)

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
      parameter == "tau_est" ~ tau_true,
      parameter == "theta_est" ~ theta_true
    )
  )
ggplot(recovery_long, aes(x = true_value, y = estimate)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~parameter, scales = "free") +
  theme_minimal() +
  labs(title = "Joint Parameter Recovery")

# Posterior SDs
ggplot(recovery_df, aes(x = factor(alpha_true), y = alpha_sd)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Posterior SD of Alpha")
