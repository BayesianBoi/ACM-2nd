// Rescorla-Wagner Reinforcement Learning Agent

// TODO
// - migrate model specification syntax "target += "
// - implement log-Likelihood

data {
  int<lower=1> n_trials;                                                        // number of trials
  array[n_trials] int<lower=0, upper=1> choice;                                 // choice made at trial t; 1 means right hand, 0 means left
  array[n_trials] int<lower=0, upper=1> opponent_choice;                        // opponent choide at trial t; 1 means right hand, 0 means left
  int<lower=1> alpha_prior_shapes;                                              // shape parameters of beta distribution, can be thought of as controlling kurtosis
  real<lower=0> tau_prior_sd;                                                   // standard deviation of lognormal prior
  real<lower=0, upper=1> initial_prob_choice;                                   // expected probability of choosing right hand on initial trial
}

parameters {
  real<lower=0, upper=1> alpha;                                                 // learning rate
  real<lower=0> tau;                                                            // inverse heat; controls stochasticity: higher → more deterministic
}

model {
  // PRIORS //
  alpha ~ beta(alpha_prior_shapes, alpha_prior_shapes);                         // learning rate prior, symmetric around 0.5
  tau ~ lognormal(0, tau_prior_sd);                                             // inverse heat prior

  // Initialize expectation
  real prob_choice = initial_prob_choice;                                       // pre-stochasticity probability of picking right hand on first trial

  for (t in 1:n_trials) {

    // Likelihood
    choice[t] ~ bernoulli_logit(tau * logit(prob_choice));                      // applying softmax to get actual righthand choice probability (special case for binary outcome)

    // Update value for next trial
    if (t < n_trials) {
      prob_choice += alpha * (opponent_choice[t] - prob_choice);                // applying Rescorla-Wagner reinforcement learning rule
    }
  }
}

generated quantities {
  // PRIOR PREDICTIVE //
  real<lower=0, upper=1> alpha_prior;
  real<lower=0> tau_prior;

  alpha_prior = beta_rng(alpha_prior_shapes, alpha_prior_shapes);               // sampling learning rate from prior distribution
  tau_prior   = lognormal_rng(0, tau_prior_sd);                                 // sampling inverse heat from prior distribution

  array[n_trials] real<lower=0, upper=1> choice_prob_priorp;                    // prior predictive (pre-stochasticity) choice probabilities
  array[n_trials] int<lower=0, upper=1> choice_priorp;                          // prior predictive of actual choices

  {                                                                             // simulating agent choices in absence of data
    real prob_choice = initial_prob_choice;

    for (t in 1:n_trials) {

      choice_prob_priorp[t] = prob_choice;
      choice_priorp[t] = bernoulli_logit_rng(tau_prior * logit(prob_choice));
    }
  }

  // POSTERIOR PREDICTIVE //
  array[n_trials] real<lower=0, upper=1> choice_prob_postp;                     // posterior predictive (pre-stochasticity) choice probabilities at each trial
  array[n_trials] int<lower=0, upper=1> choice_postp;                           // posterior predictive actual choices made

  {                                                                             // simulating agent choices after learning from the data
    real prob_choice = initial_prob_choice;

    for (t in 1:n_trials) {

      choice_prob_postp[t] = prob_choice; 
      choice_postp[t] = bernoulli_logit_rng(tau * logit(prob_choice));

      if (t < n_trials) {
        prob_choice += alpha * (opponent_choice[t] - prob_choice);
      }
    }
  }
}
