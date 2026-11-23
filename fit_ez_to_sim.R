## fit the hierarchical EZ-DDM (ez_ddm_hier.stan)
library(cmdstanr)
library(posterior)
library(dplyr)
library(tidyr)
library(ggplot2)

PROJECT_DIR <- "/Users/annegiacobello/Desktop/EZ Paper /workshop utrecht"
DATA_DIR    <- file.path(PROJECT_DIR, "data_sim")
FIT_DIR     <- file.path(PROJECT_DIR, "fits_ez")
STAN_FILE   <- file.path(PROJECT_DIR, "ez_ddm_hier.stan")
dir.create(FIT_DIR, showWarnings = FALSE, recursive = TRUE)

TEST_RUN <- FALSE              
reps     <- if (TEST_RUN) 1 else 1:10
n_warm   <- if (TEST_RUN) 500 else 1000
n_samp   <- if (TEST_RUN) 500 else 1000


true_sd <- c(sigma_nu_intercept    = 0.40,
             sigma_nu_c1           = 0.15,
             sigma_nu_c2           = 0.15,
             sigma_alpha_intercept = 0.08,
             sigma_tau_intercept   = 0.05)

mod <- cmdstan_model(STAN_FILE)


make_init <- function(n_subj, n_obs) function() list(
  beta_nu1 = 2, nu_c = 0, nu_c2 = 0, beta_alpha1 = 0, beta_tau = 0.3,
  sigma_nu_intercept = 0.3, sigma_nu_c1 = 0.1, sigma_nu_c2 = 0.1,
  sigma_alpha_intercept = 0.1, sigma_tau_intercept = 0.05,
  z = matrix(rnorm(5 * n_subj, 0, 0.1), 5, n_subj),
  sigma_nu_cell = 0.05, sigma_alpha_cell = 0.02, sigma_tau_cell = 0.01,
  z_nu_cell    = rnorm(n_obs, 0, 0.1),
  z_alpha_cell = rnorm(n_obs, 0, 0.1),
  z_tau_cell   = rnorm(n_obs, 0, 0.1)
)

pop_rows   <- list()
resid_rows <- list()
sub_rows   <- list()
cell_rows  <- list()
diag_rows  <- list()

for (r in reps) {
  rep_dir <- file.path(DATA_DIR, sprintf("rep%02d", r))
  ez      <- read.csv(file.path(rep_dir, "ez_summary.csv")) %>% arrange(subj, cell)
  true_s  <- readRDS(file.path(rep_dir, "true_subject_parameters.rds"))
  true_c  <- readRDS(file.path(rep_dir, "true_cell_parameters.rds"))
  true_p  <- readRDS(file.path(rep_dir, "true_population_parameters.rds"))

  # cells with < 2 correct responses have no defined RT variance -> set to 0, the Stan model skips them
  ez$var_rt_correct[is.na(ez$var_rt_correct)]   <- 0
  ez$mean_rt_correct[is.na(ez$mean_rt_correct)] <- 0

  stan_data <- list(
    N_obs     = nrow(ez),
    N_subj    = length(unique(ez$subj)),
    subj      = ez$subj,
    cond1     = ez$condition1,
    cond2     = ez$condition2,
    N_trials  = ez$N,
    n_correct = ez$n_correct,
    mean_rt   = ez$mean_rt_correct,
    var_rt    = ez$var_rt_correct
  )

  fit <- mod$sample(
    data = stan_data, init = make_init(stan_data$N_subj, stan_data$N_obs),
    chains = 4, parallel_chains = 4,
    iter_warmup = n_warm, iter_sampling = n_samp,
    adapt_delta = 0.9, refresh = 250, seed = 1000 + r,
    output_dir = FIT_DIR, output_basename = sprintf("ez_rep%02d", r)
  )
  fit$save_object(file.path(FIT_DIR, sprintf("ez_rep%02d.rds", r)))

  # diagnostics
  ds <- fit$diagnostic_summary(quiet = TRUE)
  diag_rows[[r]] <- data.frame(rep = r,
                               divergences = sum(ds$num_divergent),
                               max_treedepth_hits = sum(ds$num_max_treedepth),
                               max_rhat = max(fit$summary()$rhat, na.rm = TRUE),
                               min_ess_bulk = min(fit$summary()$ess_bulk, na.rm = TRUE))

  # population level
  q <- function(x) quantile(x, c(.025, .975))
  dr <- as_draws_df(fit$draws(c("beta_nu1", "nu_c", "nu_c2", "beta_alpha1", "beta_tau", names(true_sd))))
  truth_pop <- c(beta_nu1 = true_p$beta_nu1, nu_c = true_p$nu_c, nu_c2 = true_p$nu_c2,
                 beta_alpha1 = true_p$beta_alpha1, beta_tau = true_p$beta_tau, true_sd)
  pop_rows[[r]] <- bind_rows(lapply(names(truth_pop), function(p) data.frame(
    rep = r, par = p, true = truth_pop[[p]], est = mean(dr[[p]]),
    lo = q(dr[[p]])[1], hi = q(dr[[p]])[2])))


  resid_rows[[r]] <- fit$summary(c("sigma_nu_cell", "sigma_alpha_cell", "sigma_tau_cell"),
                                 mean, ~quantile(.x, c(.025, .975))) %>% mutate(rep = r)

  
  map_sub <- c(nu_intercept_s = "nu_intercept", nu_c_s = "nu_c", nu_c2_s = "nu_c2",
               alpha_s = "alpha_intercept", tau_s = "tau_intercept_raw")
  sub_rows[[r]] <- bind_rows(lapply(names(map_sub), function(p) {
    s <- fit$summary(p, mean, ~quantile(.x, c(.025, .975)))
    data.frame(rep = r, par = p, subj = true_s$subj, true = true_s[[map_sub[[p]]]],
               est = s$mean, lo = s$`2.5%`, hi = s$`97.5%`)
  }))

  # cell level (subject x cell, matched by subj + cell)
  cell_rows[[r]] <- bind_rows(lapply(c(v = "v_cell", a = "a_cell", t0 = "t0_cell"), function(p) {
    s <- fit$summary(p, mean, ~quantile(.x, c(.025, .975)))
    data.frame(subj = ez$subj, cell = ez$cell, est = s$mean, lo = s$`2.5%`, hi = s$`97.5%`)
  }), .id = "par") %>%
    left_join(true_c %>% select(subj, cell, v, a, t0) %>%
                pivot_longer(c(v, a, t0), names_to = "par", values_to = "true"),
              by = c("par", "subj", "cell")) %>%
    mutate(rep = r)
}

diag_tab  <- bind_rows(diag_rows)
pop_all   <- bind_rows(pop_rows)
resid_all <- bind_rows(resid_rows)
sub_all   <- bind_rows(sub_rows)
cell_all  <- bind_rows(cell_rows)
saveRDS(list(diag = diag_tab, pop = pop_all, resid = resid_all, sub = sub_all, cell = cell_all),
        file.path(FIT_DIR, "recovery_results.rds"))

#### recovery summaries

print(diag_tab)


print(resid_all, n = 30)

pop_all %>%
  group_by(par) %>%
  summarise(true = first(true), mean_est = mean(est),
            rel_bias_pct = 100 * (mean(est) - first(true)) / abs(first(true)),
            coverage95 = mean(lo <= true & true <= hi), .groups = "drop") %>%
  print()

rec_stats <- function(d, ...) d %>% group_by(...) %>%
  summarise(r = cor(true, est), bias = mean(est - true), rmse = sqrt(mean((est - true)^2)),
            coverage95 = mean(lo <= true & true <= hi), .groups = "drop")
rec_stats(sub_all, par)  %>% print()
rec_stats(cell_all, par) %>% print()
rec_stats(cell_all, par, cell) %>% print(n = 20)

#### plots

p_pop <- ggplot(pop_all, aes(x = factor(rep), y = est)) +
  geom_hline(aes(yintercept = true), linetype = 2, colour = "firebrick") +
  geom_pointrange(aes(ymin = lo, ymax = hi), size = 0.3) +
  facet_wrap(~ par, scales = "free_y", ncol = 5) +
  labs(x = "rep", y = "posterior mean [95% CrI]", title = "Population-level recovery") +
  theme_minimal()

p_sub <- ggplot(sub_all, aes(true, est)) +
  geom_abline(linetype = 2, colour = "firebrick") +
  geom_point(alpha = 0.3, size = 0.8) +
  facet_wrap(~ par, scales = "free", ncol = 5) +
  labs(x = "true", y = "posterior mean", title = "Subject-level recovery") +
  theme_minimal()

p_cell <- ggplot(cell_all, aes(true, est, colour = cell)) +
  geom_abline(linetype = 2) +
  geom_point(alpha = 0.3, size = 0.8) +
  facet_wrap(~ par, scales = "free") +
  labs(x = "true", y = "posterior mean", title = "Cell-level recovery (subject x cell)") +
  theme_minimal()

ggsave(file.path(FIT_DIR, "recovery_population.png"), p_pop,  width = 13, height = 5.5, dpi = 150)
ggsave(file.path(FIT_DIR, "recovery_subject.png"),    p_sub,  width = 13, height = 3.5, dpi = 150)
ggsave(file.path(FIT_DIR, "recovery_cell.png"),       p_cell, width = 11, height = 4,   dpi = 150)
