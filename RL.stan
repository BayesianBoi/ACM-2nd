// Rescorla-Wagner Reinforcement Learning Agent

// TODO
// - migrate to "target += " syntax
// - implement log-Likelihood

data {
  int<lower=1> T;                           // number of trials
  array[T] int<lower=0, upper=1> choice;    // choice made at trial t; 1 means right hand, 0 means left
  array[T] int<lower=0, upper=1> reward;    // payoff at trial t; 1 means win, 0 means loss
}

parameters {
  real<lower=0, upper=1> alpha;    // learning rate
  real<lower=0> tau;               // inverse temperature, higher → more deterministic.
  real<lower=0, upper=1> theta;    // log-odds of choosing right hand.
}

model {
  // Priors
  alpha ~ beta(2, 2);
  tau ~ lognormal(0, 1.5);
  theta ~ beta(2, 2);

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
  real<lower=0, upper=1> theta_prior;

  alpha_prior = beta_rng(1, 1);
  tau_prior   = lognormal_rng(0, 1.5);
  theta_prior = beta_rng(1, 1);

  array[T] real<lower=0, upper=1> choice_prob_priorp;
  array[T] int<lower=0, upper=1> choice_priorp;

  {
    real V_prior = theta_prior;

    for (t in 1:T) {

      choice_prob_priorp[t] = inv_logit(V_prior);
      choice_priorp[t] = bernoulli_logit_rng(tau_prior * (2 * V_prior - 1));

      if (t < T) {
        V_prior += alpha_prior * (reward[t] - V_prior);
      }
    }
  }

  // POSTERIOR PREDICTIVE
  array[T] real<lower=0, upper=1> choice_prob_postp;
  array[T] int<lower=0, upper=1> choice_postp;

  {
    real V_post = theta;

    for (t in 1:T) {

      choice_prob_postp[t] = inv_logit(V_post);
      choice_postp[t] = bernoulli_logit_rng(tau * (2 * V_post - 1));

      if (t < T) {
        V_post += alpha * (reward[t] - V_post);
      }
    }
  }
}
