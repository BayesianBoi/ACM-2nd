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
}

parameters {
  real<lower=0, upper=1> alpha;             // learning rate
  real<lower=0> tau;                        // inverse temperature, higher → more deterministic
}

model {
  // Priors
  alpha ~ beta(alpha_prior_params, alpha_prior_params);
  tau ~ lognormal(0, tau_prior_sd);

  // Unbiased initialization: no theta parameter, start from 0.5
  real V = 0.5;

  for (t in 1:T) {
    // Likelihood
    choice[t] ~ bernoulli_logit(tau * (2 * V - 1));

    // Update value for next trial
    if (t < T) {
      V += alpha * (opponent_choice[t] - V);
    }
  }
}

generated quantities {
  // PRIOR PREDICTIVE
  real<lower=0, upper=1> alpha_prior;
  real<lower=0> tau_prior;

  alpha_prior = beta_rng(alpha_prior_params, alpha_prior_params);
  tau_prior   = lognormal_rng(0, tau_prior_sd);

  array[T] real<lower=0, upper=1> choice_prob_priorp;
  array[T] int<lower=0, upper=1> choice_priorp;
  // Explicit prior-predictive environment so the agent can actually learn over trials.
  array[T] int<lower=0, upper=1> opponent_priorp;

  {
    real V = 0.5;

    for (t in 1:T) {
      // Save the trial-wise choice probability implied by current latent value V.
      choice_prob_priorp[t] = inv_logit(tau_prior * (2 * V - 1));
      choice_priorp[t] = bernoulli_rng(choice_prob_priorp[t]);
      // Prior predictive opponent is unbiased coin-flip by default.
      opponent_priorp[t] = bernoulli_rng(0.5);

      if (t < T) {
        V += alpha_prior * (opponent_priorp[t] - V);
      }
    }
  }

  // POSTERIOR PREDICTIVE
  array[T] real<lower=0, upper=1> choice_prob_postp;
  array[T] int<lower=0, upper=1> choice_postp;

  {
    real V = 0.5;

    for (t in 1:T) {
      // Store dynamic posterior predictive probability (varies with V over trials).
      choice_prob_postp[t] = inv_logit(tau * (2 * V - 1));
      choice_postp[t] = bernoulli_rng(choice_prob_postp[t]);

      if (t < T) {
        V += alpha * (opponent_choice[t] - V);
      }
    }
  }
}
