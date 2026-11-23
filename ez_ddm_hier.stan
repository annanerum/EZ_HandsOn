data {
  int<lower=1> N_obs;                          // number of subject x cell rows
  int<lower=1> N_subj;
  array[N_obs] int<lower=1, upper=N_subj> subj;
  vector[N_obs] cond1;                         // +1 = nonword, -1 = word
  vector[N_obs] cond2;                         // centred frequency (0 for nonwords)
  array[N_obs] int<lower=1> N_trials;
  array[N_obs] int<lower=0> n_correct;
  vector[N_obs] mean_rt;                       // mean RT of correct responses (s)
  vector[N_obs] var_rt;                        // variance of correct RTs (s^2)
}

transformed data {
  // RT likelihood only where it is defined (>= 2 correct responses)
  int N_rt = 0;
  for (k in 1:N_obs) if (n_correct[k] >= 2) N_rt += 1;
  array[N_rt] int idx_rt;
  {
    int j = 1;
    for (k in 1:N_obs) if (n_correct[k] >= 2) { idx_rt[j] = k; j += 1; }
  }
}

parameters {
  // population means
  real beta_nu1;        // drift intercept
  real nu_c;            // condition1 effect on drift
  real nu_c2;           // frequency effect on drift
  real beta_alpha1;     // log boundary separation
  real beta_tau;        // non-decision time (s)

  // between-subject SDs
  real<lower=0> sigma_nu_intercept;
  real<lower=0> sigma_nu_c1;
  real<lower=0> sigma_nu_c2;
  real<lower=0> sigma_alpha_intercept;
  real<lower=0> sigma_tau_intercept;

  // non-centred subject deviations (rows: nu_int, nu_c, nu_c2, alpha, tau)
  matrix[5, N_subj] z;

  // cell-level residual SDs
  real<lower=0> sigma_nu_cell;       // drift
  real<lower=0> sigma_alpha_cell;    // log boundary separation
  real<lower=0> sigma_tau_cell;      // non-decision time (s)

  // non-centred cell deviations
  vector[N_obs] z_nu_cell;
  vector[N_obs] z_alpha_cell;
  vector[N_obs] z_tau_cell;
}

transformed parameters {
  // person level
  vector[N_subj] nu_intercept_s = beta_nu1    + sigma_nu_intercept    * z[1]';
  vector[N_subj] nu_c_s         = nu_c        + sigma_nu_c1           * z[2]';
  vector[N_subj] nu_c2_s        = nu_c2       + sigma_nu_c2           * z[3]';
  vector[N_subj] alpha_s        = beta_alpha1 + sigma_alpha_intercept * z[4]';
  vector[N_subj] tau_s          = beta_tau    + sigma_tau_intercept   * z[5]';

  // cell level
  vector[N_obs] v_cell  = nu_intercept_s[subj] + nu_c_s[subj] .* cond1 + nu_c2_s[subj] .* cond2
                          + sigma_nu_cell * z_nu_cell;
  vector[N_obs] a_cell  = exp(alpha_s[subj] + sigma_alpha_cell * z_alpha_cell);
  vector[N_obs] t0_cell = tau_s[subj] + sigma_tau_cell * z_tau_cell;
}

model {
  vector[N_obs] av = a_cell .* v_cell;

  // priors (weakly informative, on the scale of a lexical decision task)
  beta_nu1    ~ normal(2, 1);
  nu_c        ~ normal(0, 1);
  nu_c2       ~ normal(0, 1);
  beta_alpha1 ~ normal(0, 0.5);       // a around 1 (0.4 - 2.7)
  beta_tau    ~ normal(0.3, 0.15);

  sigma_nu_intercept    ~ normal(0, 0.5);
  sigma_nu_c1           ~ normal(0, 0.3);
  sigma_nu_c2           ~ normal(0, 0.3);
  sigma_alpha_intercept ~ normal(0, 0.2);
  sigma_tau_intercept   ~ normal(0, 0.1);

  sigma_nu_cell    ~ normal(0, 0.3);
  sigma_alpha_cell ~ normal(0, 0.1);
  sigma_tau_cell   ~ normal(0, 0.05);

  to_vector(z) ~ std_normal();
  z_nu_cell    ~ std_normal();
  z_alpha_cell ~ std_normal();
  z_tau_cell   ~ std_normal();

  // accuracy: Pc = 1 / (1 + exp(-a v))
  n_correct ~ binomial_logit(N_trials, av);

  // EZ equations for the correct-RT moments
  {
    vector[N_rt] v_r = v_cell[idx_rt];
    vector[N_rt] a_r = a_cell[idx_rt];
    vector[N_rt] y   = exp(-av[idx_rt]);
    vector[N_rt] mrt = t0_cell[idx_rt] + (a_r ./ (2 * v_r)) .* ((1 - y) ./ (1 + y));
    vector[N_rt] vrt = (a_r ./ (2 * v_r .* v_r .* v_r))
                       .* (1 - 2 * av[idx_rt] .* y - square(y)) ./ square(1 + y);
    vector[N_rt] nc  = to_vector(n_correct[idx_rt]);

    mean_rt[idx_rt] ~ normal(mrt, sqrt(vrt ./ nc));
    var_rt[idx_rt]  ~ gamma((nc - 1) / 2, (nc - 1) ./ (2 * vrt));
  }
}
