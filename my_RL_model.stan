

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

// Now posterior predictive only
// NOT YET: Prior predictive, Log likelihood extraction, Multi-participant indexing 
generated quantities {

  array[T] int choice_rep;
  real V = theta;

  for (t in 1:T) {

    real p = inv_logit(tau * (2 * V - 1));
    choice_rep[t] = bernoulli_rng(p);

    if (t < T) {
      V = V + alpha * (reward[t] - V);
    }
  }
}


