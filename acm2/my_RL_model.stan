// RL model, one participant.
// Tracks Q1 (value of choosing 1) and Q2 (value of choosing 0) separately.
// Only the chosen action's Q is updated each trial — standard Q-learning.


data {
  int<lower=1> T;
  array[T] int<lower=0, upper=1> choice;
  array[T] int<lower=0, upper=1> reward;
}


parameters {
  real<lower=0, upper=1> alpha;   // learning rate
  real<lower=0>          tau;     // inverse temperature
  real<lower=0, upper=1> theta;   // shared starting Q for both actions
}


model {

  alpha ~ beta(2, 2);
  tau   ~ lognormal(0, 0.75);   // mostly below ~5, which is already near-deterministic over 120 trials
  theta ~ beta(2, 2);           // centered at 0.5; both Q-values start equal so p(t=1) = 0.5

  real Q1 = theta;
  real Q2 = theta;

  for (t in 1:T) {

    // standard 2-choice softmax: p(choose 1) = inv_logit(tau * (Q1 - Q2))
    real p = inv_logit(tau * (Q1 - Q2));

    choice[t] ~ bernoulli(p);

    if (t < T) {
      if (choice[t] == 1)
        Q1 += alpha * (reward[t] - Q1);
      else
        Q2 += alpha * (reward[t] - Q2);
    }
  }
}


generated quantities {

  array[T] int choice_rep;
  vector[T]    log_lik;

  real Q1 = theta;
  real Q2 = theta;

  for (t in 1:T) {

    real p = inv_logit(tau * (Q1 - Q2));

    log_lik[t]    = bernoulli_lpmf(choice[t] | p);
    choice_rep[t] = bernoulli_rng(p);

    // Q follows the observed sequence here so that log_lik[t] and choice_rep[t]
    // both condition on the same Q trajectory (trial-by-trial predictive check)
    if (t < T) {
      if (choice[t] == 1)
        Q1 += alpha * (reward[t] - Q1);
      else
        Q2 += alpha * (reward[t] - Q2);
    }
  }
}
