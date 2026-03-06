// Rescorla-Wagner Reinforcement Learning Agent

// TODO
// - migrate model specification syntax "target += "
// - implement log-Likelihood

data {
  int<lower=1> T;                           // number of trials
  array[T] int<lower=0, upper=1> choice;    // choice made at trial t; 1 means right hand, 0 means left
  array[T] int<lower=0, upper=1> reward;    // payoff at trial t; 1 means win, 0 means loss
  real alpha_prior_params;
  real tau_prior_sd;
  real theta_prior_sd;
}

parameters {
  real<lower=0, upper=1> alpha;             // learning rate
  real<lower=0> tau;                        // inverse temperature, higher → more deterministic
  real theta;                               // log-odds of choosing right hand
}

model {
  // Priors
  alpha ~ beta(alpha_prior_params, alpha_prior_params);
  tau ~ lognormal(0, tau_prior_sd);
  theta ~ normal(0, theta_prior_sd);

  // Initialize value
  real V = theta;

  for (t in 1:T) {

    // Likelihood
    choice[t] ~ bernoulli_logit(tau * (2 * V - 1));

    // Update value for next trial
    if (t < T) {
      V += alpha * (reward[t] - V);
    }
  }
}

generated quantities {
  // PRIOR PREDICTIVE
  real<lower=0, upper=1> alpha_prior;
  real<lower=0> tau_prior;
  real theta_prior;

  alpha_prior = beta_rng(alpha_prior_params, alpha_prior_params);
  tau_prior   = lognormal_rng(0, tau_prior_sd);
  theta_prior = normal_rng(0, theta_prior_sd);

  array[T] real<lower=0, upper=1> choice_prob_priorp;
  array[T] int<lower=0, upper=1> choice_priorp;

  {
    real V = theta_prior;

    for (t in 1:T) {

      choice_prob_priorp[t] = inv_logit(V);
      choice_priorp[t] = bernoulli_logit_rng(tau_prior * (2 * V - 1));

      if (t < T) {
        V += alpha_prior * (reward[t] - V);
      }
    }
  }

  // POSTERIOR PREDICTIVE
  array[T] real<lower=0, upper=1> choice_prob_postp;
  array[T] int<lower=0, upper=1> choice_postp;

  {
    real V = theta;

    for (t in 1:T) {

      choice_prob_postp[t] = inv_logit(V);
      choice_postp[t] = bernoulli_logit_rng(tau * (2 * V - 1));

      if (t < T) {
        V += alpha * (reward[t] - V);
      }
    }
  }
}
