

// RL only, one participant


// DATA BLOCK
data {
  int<lower=1> T;                  // number of trials
  array[T] int<lower=0, upper=1> choice;
  array[T] int<lower=0, upper=1> reward;
}


// PARAMETERS
// for now, Fix V₁ = 0.5
// infer theta
parameters {
  real<lower=0, upper=1> alpha;    // learning rate
  real<lower=0> tau;               // inverse temperature
  real<lower=0, upper=1> theta;    // initial value V1 - Early trials become informative.
}

// ------------------------------------------------------------


// MODEL BLOCK

model {

  // Priors
  alpha ~ beta(2, 2);
  tau ~ lognormal(0, 1);
  theta ~ beta(2, 2);

  // Initialize value
  real V = theta;

  for (t in 1:T) {

    // Map V from [0,1] to [-1,1]
    real p = inv_logit(tau * (2 * V - 1)); // Higher TAU → more deterministic.

    // Likelihood
    choice[t] ~ bernoulli(p);

    // Update value for next trial
    if (t < T) {
      V = V + alpha * (reward[t] - V);
    }
  }
}

generated quantities {

  // ============================
  // PRIOR PREDICTIVE
  // ============================

  real<lower=0, upper=1> alpha_prior;
  real tau_prior;
  real<lower=0, upper=1> theta_prior;

  alpha_prior = beta_rng(1, 1);
  tau_prior   = normal_rng(0, 5);
  theta_prior = beta_rng(1, 1);

  array[T] int<lower=0, upper=1> prior_preds;

  {
    real V_prior = theta_prior;

    for (t in 1:T) {

      real logit_p_prior = tau_prior * (2 * V_prior - 1);
      real p_prior = inv_logit(logit_p_prior);

      prior_preds[t] = bernoulli_rng(p_prior);

      if (t < T) {
        V_prior = V_prior + alpha_prior * (reward[t] - V_prior);
      }
    }
  }


  // ============================
  // POSTERIOR PREDICTIVE
  // ============================

  array[T] int<lower=0, upper=1> posterior_preds;

  {
    real V_post = theta;

    for (t in 1:T) {

      real logit_p_post = tau * (2 * V_post - 1);
      real p_post = inv_logit(logit_p_post);

      posterior_preds[t] = bernoulli_rng(p_post);

      if (t < T) {
        V_post = V_post + alpha * (reward[t] - V_post);
      }
    }
  }
}


