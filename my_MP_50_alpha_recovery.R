

# recovery using the MP algorithm - 50/50 rate opponent
# theta and tau is constant, only alpha varies
set.seed(123)
# --------------------------------------
#  1. MODEL
# --------------------------------------
library(cmdstanr)
library(posterior)
library(dplyr)
library(ggplot2)

model <- cmdstan_model("my_RL_model.stan")

T <- 120
rate_opponent <- 0.5

# --------------------------------------
#  2. PRIOR PREDICTIVE CHECK
# --------------------------------------
# dummy data (ignored in prior predictive because priors dominate)
dummy_data <- list(
  T = T,
  choice = rep(0, T),
  reward = rep(0, T)
)

fit_prior <- model$sample(
  data = dummy_data,
  seed = 1234,
  chains = 4,
  iter_sampling = 500,
  iter_warmup = 0,
  fixed_param = TRUE
)


draws_prior <- fit_prior$draws("choice_rep")
# Convert to draws_array
draws_array <- as_draws_array(draws_prior)
# Combine chains
draws_matrix <- posterior::as_draws_matrix(draws_array)
# Extract only columns for choice_rep
choice_cols <- grep("choice_rep", colnames(draws_matrix))
prior_choices <- draws_matrix[, choice_cols]

# average across posterior draws
prior_mean_choice <- colMeans(prior_choices)

plot(1:T, prior_mean_choice, type = "l",
     ylim = c(0,1),
     main = "Prior Predictive Mean Choice",
     ylab = "P(choice=1)",
     xlab = "Trial")
abline(h = 0.5, lty = 2)

# --------------------------------------
#  3. POSTERIOR PREDICTIVE CHECK
# --------------------------------------
# SIMULATE data using the MP algorithm with a 50/50 opponent
simulate_rl_mp <- function(alpha, tau, theta, T, rate_opponent = 0.5) {
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
    if (t < T)
      V[t+1] <- V[t] + alpha * (reward[t] - V[t])
  }
  
  list(choice = choice, reward = reward)
}

sim_data <- simulate_rl_mp(alpha = 0.6, tau = 3, theta = 0.5, T = T)

data_list <- list(
  T = T,
  choice = sim_data$choice,
  reward = sim_data$reward
)

fit_post <- model$sample(
  data = data_list,
  seed = 1234,
  chains = 4,
  iter_sampling = 1000,
  iter_warmup = 1000
)

# Posterior predictive: 
draws_post <- as_draws_matrix(fit_post$draws("choice_rep"))
choice_cols <- grep("choice_rep", colnames(draws_post))
post_rep <- draws_post[, choice_cols]

# mean across draws for each trial
post_choice_means <- rowMeans(post_rep)

hist(post_choice_means, breaks = 30,
     main = "Posterior Predictive Overall Choice Rate",
     xlab = "Mean P(choice=1)")
abline(v = mean(sim_data$choice), col = "red", lwd = 2)

# --------------------------------------
#  4. PARAMETER RECOVERY
# --------------------------------------

# RL agent = matcher (wins if same choice as opponent)
# Opponent = random with bias rate_opponent
# Reward[t] = 1 if agent_choice == opponent_choice else 0

alpha_grid <- c(0.1, 0.3, 0.6, 0.9)
tau_true <- 3
theta_true <- 0.5
n_reps <- 10

recovery_results <- list()
counter <- 1

for (alpha_true in alpha_grid) {
  for (rep in 1:n_reps) {
    
    sim <- simulate_rl_mp(
      alpha = alpha_true,
      tau = tau_true,
      theta = theta_true,
      T = T,
      rate_opponent = 0.5
    )
    
    data_list <- list(
      T = T,
      choice = sim$choice,
      reward = sim$reward
    )
    
    fit <- model$sample(
      data = data_list,
      chains = 4,
      iter_sampling = 1000,
      iter_warmup = 1000,
      refresh = 0
    )
    
    draws <- as_draws_df(fit$draws())
    
    recovery_results[[counter]] <- data.frame(
      alpha_true = alpha_true,
      alpha_est = mean(draws$alpha),
      alpha_sd = sd(draws$alpha)
    )
    
    counter <- counter + 1
  }
}

recovery_df <- bind_rows(recovery_results)


# RECOV. PLOTS ----------------------------------------------------------------------------------

# Alpha Recovery Plot
ggplot(recovery_df, aes(x = alpha_true, y = alpha_est)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  theme_minimal() +
  labs(title = "Parameter Recovery: Alpha")

# Posterior SD
ggplot(recovery_df, aes(x = factor(alpha_true), y = alpha_sd)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Posterior SD of Alpha")
