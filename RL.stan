// Rescorla-Wagner Reinforcement Learning Agent

// TODO
// - migrate model specification syntax "target += "
// - implement log-Likelihood

data {
  int<lower=1> n_trials;                                       // number of trials
  array[n_trials] int<lower=0, upper=1> choice;                // choice made at trial t; 1 means right hand, 0 means left
  array[n_trials] int<lower=0, upper=1> opponent_choice;       // payoff at trial t; 1 means win, 0 means loss
  int<lower=1> alpha_prior_params;
  real<lower=0> tau_prior_sd;
  real<lower=0, upper=1> initial_expected_prob;
}

parameters {
  real<lower=0, upper=1> alpha;                         // learning rate
  real<lower=0> tau;                                    // inverse temperature, higher → more deterministic
}

model {
  // Priors
  alpha ~ beta(alpha_prior_params, alpha_prior_params);
  tau ~ lognormal(0, tau_prior_sd);

  // Initialize expectation
  real expected_prob = initial_expected_prob;

  for (t in 1:n_trials) {

    // Likelihood
    choice[t] ~ bernoulli_logit(tau * logit(expected_prob));

    // Update value for next trial
    if (t < n_trials) {
      expected_prob += alpha * (opponent_choice[t] - expected_prob);
    }
  }
}

generated quantities {
  // PRIOR PREDICTIVE
  real<lower=0, upper=1> alpha_prior;
  real<lower=0> tau_prior;

  alpha_prior = beta_rng(alpha_prior_params, alpha_prior_params);
  tau_prior   = lognormal_rng(0, tau_prior_sd);

  array[n_trials] real<lower=0, upper=1> choice_prob_priorp;
  array[n_trials] int<lower=0, upper=1> choice_priorp;

  {
    real expected_prob = initial_expected_prob;

    for (t in 1:n_trials) {

      choice_prob_priorp[t] = expected_prob;
      choice_priorp[t] = bernoulli_logit_rng(tau_prior * logit(expected_prob));
    }
  }

  // POSTERIOR PREDICTIVE
  array[n_trials] real<lower=0, upper=1> choice_prob_postp;
  array[n_trials] int<lower=0, upper=1> choice_postp;

  {
    real expected_prob = initial_expected_prob;

    for (t in 1:n_trials) {

      choice_prob_postp[t] = expected_prob;
      choice_postp[t] = bernoulli_logit_rng(tau * logit(expected_prob));

      if (t < n_trials) {
        expected_prob += alpha * (opponent_choice[t] - expected_prob);
      }
    }
  }
}
