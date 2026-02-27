set.seed(123)

library(cmdstanr)
library(posterior)
library(dplyr)
library(ggplot2)

model <- cmdstan_model("my_RL_model.stan")

T             <- 120
rate_opponent <- 0.5


# --------------------------------------
#  SIMULATOR
# --------------------------------------
# Tracks Q1 and Q2 separately; only the chosen action's Q is updated.
# Must match the Stan model exactly or simulation and inference diverge.
simulate_rl_mp <- function(alpha, tau, theta, T, rate_opponent = 0.5) {
  Q1     <- theta    # value of choosing 1
  Q2     <- theta    # value of choosing 0; equal start means p(t=1) = 0.5
  choice <- integer(T)
  reward <- integer(T)

  for (t in seq_len(T)) {
    p         <- plogis(tau * (Q1 - Q2))
    choice[t] <- rbinom(1, 1, p)
    opponent  <- rbinom(1, 1, rate_opponent)
    reward[t] <- as.integer(choice[t] == opponent)

    if (t < T) {
      if (choice[t] == 1)
        Q1 <- Q1 + alpha * (reward[t] - Q1)
      else
        Q2 <- Q2 + alpha * (reward[t] - Q2)
    }
  }

  list(choice = choice, reward = reward)
}


# --------------------------------------
#  2. PRIOR PREDICTIVE CHECK
# --------------------------------------
# Stan always conditions on the data you pass in — there is no clean way
# to skip the likelihood without restructuring the model. Sampling from
# the priors directly in R is simpler and actually correct.

n_prior              <- 2000
prior_choice_matrix  <- matrix(NA_integer_, nrow = n_prior, ncol = T)

for (i in seq_len(n_prior)) {
  a  <- rbeta(1, 2, 2)
  ta <- rlnorm(1, 0, 0.75)
  th <- rbeta(1, 2, 2)
  s  <- simulate_rl_mp(alpha = a, tau = ta, theta = th, T = T)
  prior_choice_matrix[i, ] <- s$choice
}

prior_df <- data.frame(
  trial = seq_len(T),
  mean  = colMeans(prior_choice_matrix),
  lo    = apply(prior_choice_matrix, 2, quantile, 0.05),
  hi    = apply(prior_choice_matrix, 2, quantile, 0.95)
)

prior_pred_plot <- ggplot(prior_df, aes(x = trial)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.2) +
  geom_line(aes(y = mean)) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  ylim(0, 1) +
  labs(title = "Prior Predictive: Mean Choice (90% interval)",
       y = "P(choice=1)", x = "Trial") +
  theme_minimal()

ggsave("MP_prior_pred.png", prior_pred_plot, width = 6, height = 4)


# --------------------------------------
#  3. POSTERIOR PREDICTIVE CHECK
# --------------------------------------
sim_data <- simulate_rl_mp(alpha = 0.6, tau = 3, theta = 0.5, T = T)

data_list <- list(
  T      = T,
  choice = sim_data$choice,
  reward = sim_data$reward
)

fit_post <- model$sample(
  data          = data_list,
  seed          = 1234,
  chains        = 4,
  iter_sampling = 2000,
  iter_warmup   = 1000
)

# divergences can pass R-hat/ESS undetected
fit_post$diagnostic_summary()

draws_post        <- as_draws_matrix(fit_post$draws("choice_rep"))
choice_cols       <- grep("choice_rep", colnames(draws_post))
post_rep          <- draws_post[, choice_cols]
post_choice_means <- colMeans(post_rep)

post_pred_plot <- ggplot(data.frame(x = post_choice_means), aes(x = x)) +
  geom_histogram(bins = 30, fill = "grey70", color = "white") +
  geom_vline(xintercept = mean(sim_data$choice), color = "red", linewidth = 1) +
  labs(title = "Posterior Predictive: Overall Choice Rate",
       x = "Mean P(choice=1)", y = "Count") +
  theme_minimal()

ggsave("MP_post_pred.png", post_pred_plot, width = 6, height = 4)


# --------------------------------------
#  4. PARAMETER RECOVERY
# --------------------------------------
# All three parameters now vary — in the original only alpha was varied,
# so tau and theta recovery was never actually tested.
# With a 50/50 opponent, Q1 and Q2 both drift toward ~0.5 regardless of
# alpha, so high alpha is hardest to recover (fast convergence, little signal).

alpha_grid <- c(0.1, 0.3, 0.6, 0.9)
tau_grid   <- c(1, 3, 6)
theta_grid <- c(0.2, 0.5, 0.8)
n_reps     <- 5

recovery_results <- list()
counter          <- 1

for (alpha_true in alpha_grid) {
  for (tau_true in tau_grid) {
    for (theta_true in theta_grid) {
      for (rep in seq_len(n_reps)) {

        sim <- simulate_rl_mp(
          alpha         = alpha_true,
          tau           = tau_true,
          theta         = theta_true,
          T             = T,
          rate_opponent = rate_opponent
        )

        data_list <- list(
          T      = T,
          choice = sim$choice,
          reward = sim$reward
        )

        fit <- model$sample(
          data          = data_list,
          seed          = 1000 + counter,   # one reproducible seed per iteration
          chains        = 4,
          iter_sampling = 2000,
          iter_warmup   = 1000,
          refresh       = 0
        )

        # divergences are extracted directly from sampler output to avoid printing
        sampler_diag <- fit$sampler_diagnostics(inc_warmup = FALSE)
        n_divergent  <- sum(sampler_diag[,, "divergent__"])

        param_summary <- fit$summary(variables = c("alpha", "tau", "theta"))
        max_rhat      <- max(param_summary$rhat,     na.rm = TRUE)
        min_ess_bulk  <- min(param_summary$ess_bulk, na.rm = TRUE)

        # divergences can indicate geometry problems even when R-hat looks fine
        converged <- (max_rhat < 1.01) && (min_ess_bulk > 400) && (n_divergent == 0)

        draws <- as_draws_df(fit$draws(variables = c("alpha", "tau", "theta")))

        # 95% CI: more honest than +-1 SD when the posterior is skewed near boundaries
        recovery_results[[counter]] <- data.frame(
          alpha_true   = alpha_true,
          alpha_est    = mean(draws$alpha),
          alpha_lo     = quantile(draws$alpha, 0.025),
          alpha_hi     = quantile(draws$alpha, 0.975),

          tau_true     = tau_true,
          tau_est      = mean(draws$tau),
          tau_lo       = quantile(draws$tau, 0.025),
          tau_hi       = quantile(draws$tau, 0.975),

          theta_true   = theta_true,
          theta_est    = mean(draws$theta),
          theta_lo     = quantile(draws$theta, 0.025),
          theta_hi     = quantile(draws$theta, 0.975),

          max_rhat     = max_rhat,
          min_ess_bulk = min_ess_bulk,
          n_divergent  = n_divergent,
          converged    = converged
        )

        counter <- counter + 1
      }
    }
  }
}

recovery_df <- bind_rows(recovery_results)


# RECOVERY PLOTS ---------------------------------------------------------------
# Each plot pools across all combinations of the other two parameters.
# Error bars are 95% credible intervals.

alpha_plot <- ggplot(recovery_df, aes(x = alpha_true, y = alpha_est)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_errorbar(aes(ymin = alpha_lo, ymax = alpha_hi),
                width = 0.02, alpha = 0.4) +
  geom_point() +
  theme_minimal() +
  labs(title = "Parameter Recovery: Alpha",
       x = "True alpha", y = "Posterior mean")

tau_plot <- ggplot(recovery_df, aes(x = tau_true, y = tau_est)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_errorbar(aes(ymin = tau_lo, ymax = tau_hi),
                width = 0.1, alpha = 0.4) +
  geom_point() +
  theme_minimal() +
  labs(title = "Parameter Recovery: Tau",
       x = "True tau", y = "Posterior mean")

theta_plot <- ggplot(recovery_df, aes(x = theta_true, y = theta_est)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_errorbar(aes(ymin = theta_lo, ymax = theta_hi),
                width = 0.02, alpha = 0.4) +
  geom_point() +
  theme_minimal() +
  labs(title = "Parameter Recovery: Theta",
       x = "True theta", y = "Posterior mean")

# CI width across alpha levels shows how uncertainty scales with the true value
alpha_sd_plot <- ggplot(recovery_df,
                        aes(x = factor(alpha_true), y = alpha_hi - alpha_lo)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "95% CI Width: Alpha",
       x = "True alpha", y = "CI width")

ggsave("alpha_recovery_plot.png", alpha_plot,    width = 6, height = 4)
ggsave("tau_recovery_plot.png",   tau_plot,      width = 6, height = 4)
ggsave("theta_recovery_plot.png", theta_plot,    width = 6, height = 4)
ggsave("alpha_sd_plot.png",       alpha_sd_plot, width = 6, height = 4)
