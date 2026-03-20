// Rescorla-Wagner Reinforcement Learning Agent

// TODO
// - migrate model specification syntax "target += "

data {
  int<lower=1> n_trials;                                      // number of trials
  array[n_trials] int<lower=0, upper=1> choice;               // choice made at trial t; 1 means right hand, 0 means left
  array[n_trials] int<lower=0, upper=1> opponent_choice;      // opponent choice at trial t; 1 means right hand, 0 means left
  int<lower=1> alpha_prior_shapes;                            // shape parameters of beta distribution
  real<lower=0> tau_prior_sd;                                 // standard deviation of lognormal prior
  real<lower=0, upper=1> initial_prob_choice;                 // expected probability of choosing right hand on initial trial
}

parameters {
  real<lower=0, upper=1> alpha;                               // learning rate
  real<lower=0> tau;                                          // inverse heat; higher means more deterministic responding
}

model {
  // PRIORS //
  alpha ~ beta(alpha_prior_shapes, alpha_prior_shapes);
  tau ~ lognormal(0, tau_prior_sd);

  real prob_choice = initial_prob_choice;

  for (t in 1:n_trials) {
    choice[t] ~ bernoulli_logit(tau * logit(prob_choice));

    if (t < n_trials) {
      prob_choice += alpha * (opponent_choice[t] - prob_choice);
    }
  }
}

generated quantities {
  // PRIOR PREDICTIVE //
  real<lower=0, upper=1> alpha_prior;
  real<lower=0> tau_prior;

  alpha_prior = beta_rng(alpha_prior_shapes, alpha_prior_shapes);
  tau_prior   = lognormal_rng(0, tau_prior_sd);

  array[n_trials] real<lower=0, upper=1> choice_prob_priorp;  // prior predictive belief trajectory
  array[n_trials] int<lower=0, upper=1> choice_priorp;        // prior predictive actual choices

  {
    // Prior predictive still means alpha/tau come from the prior; the agent can
    // still update over the provided opponent sequence because this is a learning model.
    real prob_choice = initial_prob_choice;

    for (t in 1:n_trials) {
      choice_prob_priorp[t] = prob_choice;
      choice_priorp[t] = bernoulli_logit_rng(tau_prior * logit(prob_choice));

      if (t < n_trials) {
        prob_choice += alpha_prior * (opponent_choice[t] - prob_choice);
      }
    }
  }

  // POSTERIOR PREDICTIVE //
  array[n_trials] real<lower=0, upper=1> choice_prob_postp;   // posterior predictive belief trajectory
  array[n_trials] int<lower=0, upper=1> choice_postp;         // posterior predictive actual choices

  {
    real prob_choice = initial_prob_choice;

    for (t in 1:n_trials) {
      choice_prob_postp[t] = prob_choice;
      choice_postp[t] = bernoulli_logit_rng(tau * logit(prob_choice));

      if (t < n_trials) {
        prob_choice += alpha * (opponent_choice[t] - prob_choice);
      }
    }
  }

  // LOG PRIOR (for priorsense) //
  real lprior;
  lprior = beta_lpdf(alpha | alpha_prior_shapes, alpha_prior_shapes) +
    lognormal_lpdf(tau | 0, tau_prior_sd);

  // LOG-LIKELIHOOD //
  array[n_trials] real log_lik;

  {
    real prob_choice = initial_prob_choice;

    for (t in 1:n_trials) {
      log_lik[t] = bernoulli_logit_lpmf(choice[t] | tau * logit(prob_choice));

      if (t < n_trials) {
        prob_choice += alpha * (opponent_choice[t] - prob_choice);
      }
    }
  }
}
