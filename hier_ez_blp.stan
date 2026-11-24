// Each cell (participant x condition) is summarised by:
// number correct responses, mean RT of correct responses, variance of correct RTs
// The EZ equations link them to 3 diffusion parameters:
// v = drift rate (speed of evidence accumulation)
// a = boundary separation (response caution)
// t0 = non-decision time (encoding + motor, in seconds)
// scaling: s = 1 (classic EZ uses s = 0.1 --> multiply our v and a by 0.1 to compare)

data {
  int<lower=1> N_cells;                       // number of cells (participant x condition)
  int<lower=1> J;                             // number of participants
  int<lower=1> K;                             // number of drift coefficients (columns of X)
  array[N_cells] int<lower=1, upper=J> subj;  // participant of each cell
  matrix[N_cells, K] X;                       // design matrix: condition coding of each cell

  // the data: EZ summary statistics per cell
  array[N_cells] int<lower=1> n_trials;       // number of trials
  array[N_cells] int<lower=2> n_correct;      // number of correct responses
  vector[N_cells] mean_rt;                    // mean RT of correct responses (s)
  vector[N_cells] var_rt;                     // variance of correct RTs (s^2)

  int<lower=0, upper=1> prior_only;           // 1 = ignore the data (prior predictive check)
}

parameters {
  // population level
  vector[K] beta_v;              // average drift coefficients (intercept, lex, freq)
  vector<lower=0>[K] sd_v;       // how much participants differ in each coefficient
  real mu_log_a;                 // average log boundary
  real<lower=0> sd_log_a;        // how much participants differ in log boundary
  real mu_log_t0;                // average log non-decision time
  real<lower=0> sd_log_t0;       // how much participants differ in log non-decision time

  // participant level, as z-scores (standardised deviations from the average)
  matrix[J, K] z_v;
  vector[J] z_a;
  vector[J] z_t0;
}

transformed parameters {
  // participant parameters = average + SD * z-score
  // (the same as b_v ~ normal(beta_v, sd_v) etc., but written so that the
  //  sampler works much better: "non-centred parameterisation")
  matrix[J, K] b_v;              // drift coefficients of each participant
  for (k in 1:K) {
    b_v[, k] = beta_v[k] + sd_v[k] * z_v[, k];
  }
  vector[J] a  = exp(mu_log_a  + sd_log_a  * z_a);    // boundary of each participant (> 0)
  vector[J] t0 = exp(mu_log_t0 + sd_log_t0 * z_t0);   // non-decision time of each participant (> 0)

  // EZ predictions for every cell
  vector[N_cells] v;             // drift rate
  vector[N_cells] Pc;            // predicted probability correct
  vector[N_cells] MRT;           // predicted mean RT
  vector[N_cells] VRT;           // predicted RT variance

  for (i in 1:N_cells) {
    int j = subj[i];                    // participant of this cell
    v[i] = X[i] * b_v[j]';              // drift = condition coding x participant's coefficients
    real y = a[j] * v[i];

    // EZ equations but written with tanh/sinh/cosh so numerically stable
    Pc[i] = inv_logit(y);                                          // = 1 / (1 + exp(-a v))
    if (abs(y) < 1e-4) {
      // drift ~ 0: the formulas below would divide by ~0, so we use their limits
      MRT[i] = t0[j] + a[j]^2 / 4;
      VRT[i] = a[j]^4 / 24;
    } else {
      MRT[i] = t0[j] + a[j] / (2 * v[i]) * tanh(y / 2);
      VRT[i] = a[j] / (2 * v[i]^3) * (sinh(y) - y) / (cosh(y) + 1);
    }
  }
}

model {
  // priors for the population level
  beta_v[1] ~ normal(2, 1);            // average non-worddrift: clearly above 0
  beta_v[2:K] ~ normal(0, 1);          // condition effects: centred on "no effect"
  sd_v ~ normal(0, 1);
  mu_log_a ~ normal(log(1.5), 0.5);    // median a ~1.5
  sd_log_a ~ normal(0, 0.5);
  mu_log_t0 ~ normal(log(0.35), 0.3);  // median t0 ~350 ms
  sd_log_t0 ~ normal(0, 0.3);

  // participants come from the population (the "hierarchical" part)
  // like (1 + lex + freq || subj) in lme4: independent random effects
  to_vector(z_v) ~ std_normal();
  z_a  ~ std_normal();
  z_t0 ~ std_normal();

  // likelihood: how the observed summary statistics scatter around the EZ predictions
  if (prior_only == 0) {
    vector[N_cells] n_c = to_vector(n_correct);
    n_correct ~ binomial(n_trials, Pc);                       // accuracy
    mean_rt   ~ normal(MRT, sqrt(VRT ./ n_c));                // mean of n RTs (central limit theorem)
    var_rt    ~ normal(VRT, VRT .* sqrt(2 ./ (n_c - 1)));     // variance of n RTs
  }
}

generated quantities {
  // simulated data for the posterior predictive check:
  // "if the model is right, what data would we expect?"
  array[N_cells] int pred_n_correct = binomial_rng(n_trials, Pc);
  vector[N_cells] pred_mean_rt;
  vector[N_cells] pred_var_rt;
  for (i in 1:N_cells) {
    pred_mean_rt[i] = normal_rng(MRT[i], sqrt(VRT[i] / n_correct[i]));
    pred_var_rt[i]  = normal_rng(VRT[i], VRT[i] * sqrt(2.0 / (n_correct[i] - 1)));
  }
}
