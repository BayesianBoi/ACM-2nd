// Rescorla-Wagner Reinforcement Learning Agent

// TODO
// - migrate model specification syntax "target += "
// - implement log-Likelihood

data {
  int<lower=1> T;                           // number of trials
  array[T] int<lower=0, upper=1> choice;    // choice made at trial t; 1 means right hand, 0 means left
  array[T] int<lower=0, upper=1> opponent_choice;    // opponent action at trial t; 1 means right, 0 means left
  int<lower=1> alpha_prior_params;
  real<lower=0> tau_prior_sd;
  real<lower=0> theta_prior_sd;
}

parameters {
  real<lower=0, upper=1> alpha;             // learning rate
  real<lower=0> tau;                        // inverse temperature, higher → more deterministic
  real theta_logit;                         // log-odds of choosing right hand
}

transformed parameters
{
    // Keep theta unconstrained for sampling, then map once to probability scale for RW updates.
    real theta_prob = inv_logit(theta_logit);
}

model {
  // Priors
  alpha ~ beta(alpha_prior_params, alpha_prior_params);
  tau ~ lognormal(0, tau_prior_sd);
  theta_logit ~ normal(0, theta_prior_sd);

  // Initialize value
  real V = theta_prob;

  for (t in 1:T) {
    // Task feedback is "did we match the opponent?" (win/loss), not raw opponent action.
    int reward_t = (choice[t] == opponent_choice[t]);

    // Likelihood
    choice[t] ~ bernoulli_logit(tau * (2 * V - 1));

    // Update value for next trial
    if (t < T) {
      // RW update must pull toward current prediction error around V (not around fixed theta_prob).
      V += alpha * (reward_t - V);
    }
  }
}

generated quantities {
  // PRIOR PREDICTIVE
  real<lower=0, upper=1> alpha_prior;
  real<lower=0> tau_prior;
  real theta_prior_logit;
  real theta_prior_prob;

  alpha_prior = beta_rng(alpha_prior_params, alpha_prior_params);
  tau_prior   = lognormal_rng(0, tau_prior_sd);
  theta_prior_logit = normal_rng(0, theta_prior_sd);
  theta_prior_prob = inv_logit(theta_prior_logit);

  array[T] real<lower=0, upper=1> choice_prob_priorp;
  array[T] int<lower=0, upper=1> choice_priorp;
  // Explicit prior-predictive environment so the agent can actually learn over trials.
  array[T] int<lower=0, upper=1> opponent_priorp;
  array[T] int<lower=0, upper=1> reward_priorp;

  {
    real V = theta_prior_prob;

    for (t in 1:T) {
      // Save the trial-wise choice probability implied by current latent value V.
      choice_prob_priorp[t] = inv_logit(tau_prior * (2 * V - 1));
      choice_priorp[t] = bernoulli_rng(choice_prob_priorp[t]);
      // Prior predictive opponent is unbiased coin-flip by default.
      opponent_priorp[t] = bernoulli_rng(0.5);
      reward_priorp[t] = (choice_priorp[t] == opponent_priorp[t]);

      if (t < T) {
        // Trial-by-trial prior predictive learning dynamics.
        V += alpha_prior * (reward_priorp[t] - V);
      }
    }
  }

  // POSTERIOR PREDICTIVE
  array[T] real<lower=0, upper=1> choice_prob_postp;
  array[T] int<lower=0, upper=1> choice_postp;
  // Posterior predictive reward reconstructed from sampled choice vs observed opponent action.
  array[T] int<lower=0, upper=1> reward_postp;

  {
    real V = theta_prob;

    for (t in 1:T) {
      // Store dynamic posterior predictive probability (varies with V over trials).
      choice_prob_postp[t] = inv_logit(tau * (2 * V - 1));
      choice_postp[t] = bernoulli_rng(choice_prob_postp[t]);
      reward_postp[t] = (choice_postp[t] == opponent_choice[t]);

      if (t < T) {
        V += alpha * (reward_postp[t] - V);
      }
    }
  }
}
