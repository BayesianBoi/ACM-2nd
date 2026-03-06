// Rescorla-Wagner Reinforcement Learning Agent

// TODO
// - migrate model specification syntax "target += "
// - implement log-Likelihood

data {
  int<lower=1> T;                           // number of trials
  array[T] int<lower=0, upper=1> choice;    // choice made at trial t; 1 means right hand, 0 means left
  array[T] int<lower=0, upper=1> opponent_choice;    // payoff at trial t; 1 means win, 0 means loss
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
    real theta_prob = inv_logit(theta_logit);
}

model {
  // Priors
  alpha ~ beta(alpha_prior_params, alpha_prior_params);
  tau ~ lognormal(0, tau_prior_sd);
  theta_logit ~ normal(0, theta_prior_sd);

  // Initialize value
  real V = theta_prob;
  real choice_prob;

  for (t in 1:T) {

    // Likelihood
    choice[t] ~ bernoulli_logit(tau * (2 * V - 1));

    // Update value for next trial
    if (t < T) {
      V += alpha * (opponent_choice[t] - theta_prob);
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

  {
    real V = theta_prior_prob;

    for (t in 1:T) {

      choice_priorp[t] = bernoulli_logit_rng(tau_prior * (2 * V - 1));
      choice_prob_priorp[t] = theta_prior_prob;
    }
  }

  // POSTERIOR PREDICTIVE
  array[T] real<lower=0, upper=1> choice_prob_postp;
  array[T] int<lower=0, upper=1> choice_postp;

  {
    real V = theta_prob;

    for (t in 1:T) {

      choice_postp[t] = bernoulli_logit_rng(tau * (2 * V - 1));
      choice_prob_postp[t] = theta_prob;

      if (t < T) {
        V += alpha * (opponent_choice[t] - theta_prob);
      }
    }
  }
}
