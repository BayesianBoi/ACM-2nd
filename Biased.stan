// Biased Agent Model
// Fixed choice bias (theta), no learning from opponent.

data {
  int<lower=1> n_trials;
  array[n_trials] int<lower=0, upper=1> choice;
  real<lower=0> theta_prior_shape1;                               // Beta prior shape 1 for theta
  real<lower=0> theta_prior_shape2;                               // Beta prior shape 2 for theta
}

parameters {
  real<lower=0, upper=1> theta;                                   // fixed bias toward choice = 1
}

model {
  theta ~ beta(theta_prior_shape1, theta_prior_shape2);

  for (t in 1:n_trials) {
    choice[t] ~ bernoulli(theta);
  }
}

generated quantities {
  // PRIOR PREDICTIVE //
  real<lower=0, upper=1> theta_prior;
  theta_prior = beta_rng(theta_prior_shape1, theta_prior_shape2);

  array[n_trials] int<lower=0, upper=1> choice_priorp;
  array[n_trials] int<lower=0, upper=1> choice_postp;

  for (t in 1:n_trials) {
    choice_priorp[t] = bernoulli_rng(theta_prior);
    choice_postp[t]  = bernoulli_rng(theta);
  }

  // LOG PRIOR (for priorsense) //
  real lprior;
  lprior = beta_lpdf(theta | theta_prior_shape1, theta_prior_shape2);

  // LOG-LIKELIHOOD //
  array[n_trials] real log_lik;

  for (t in 1:n_trials) {
    log_lik[t] = bernoulli_lpmf(choice[t] | theta);
  }
}