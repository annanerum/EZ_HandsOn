# Keuleers E, Lacey P, Rastle K, Brysbaert M (2012). 
# “The British Lexicon Project: Lexical decision data for 28,730 monosyllabic
# and disyllabic English words.” Behavior research methods, 44(1), 287–304.
# R package `bcogsci` 
# model: hier_ez_blp.stan


# install.packages(c("remotes", "dplyr", "tidyr", "ggplot2", "posterior", "bayesplot"))
# install.packages("cmdstanr",
#                  repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
# cmdstanr::check_cmdstan_toolchain()   # C++ compiler available?
# cmdstanr::install_cmdstan()           # downloads & builds CmdStan 
#remotes::install_github("bnicenboim/bcogsci")   # data 

library(dplyr)
library(tidyr)
library(ggplot2)
library(cmdstanr)
library(posterior)
library(bayesplot)

theme_set(theme_bw(base_size = 13))
set.seed(2026)

# set the working directory to the folder this script is saved in
# so R finds hier_ez_blp.stan and creates the fits folder there
# (solution for RStudio, otherwise: Session > Set Working Directory > To Source File Location)
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}
getwd()   # should be the folder where you saved this script and hier_ez_blp.stan

# stop if Stan file is not where script is
if (!file.exists("hier_ez_blp.stan")) {
  stop("hier_ez_blp.stan not found in ", getwd())
}

# folder for the fits
dir.create("fits", showWarnings = FALSE)


# download data
tmp <- file.path(tempdir(), "df_blp.rda")
download.file("https://github.com/bnicenboim/bcogsci/raw/master/data/df_blp.rda",
              destfile = tmp, mode = "wb")
load(tmp)

# for the workshop we use the smaller subset `df_blp`
# note: the help page ?df_blp says 40 participant but the current data have
# 20 participants x 1200 trials, 600 words + 600 nonwords each


# looking at data
glimpse(df_blp)
# subj   participant id
# lex    "word" or "non-word"
# string the letter string shown
# acc    1 = correct, 0 = error
# rt     response time in ms
# freq   word frequency, 0 for nonwords

df_blp |> count(subj)          # trials per participant
df_blp |> count(subj, lex)     # 600 words, 600 nonwords

# accuracy and mean RT for word and non-word
df_blp |>
  group_by(lex) |>
  summarise(accuracy = mean(acc), mean_rt = mean(rt), median_rt = median(rt))

# RT distribution
# note the long right tail (max is > 45 s)
summary(df_blp$rt)
ggplot(df_blp, aes(rt)) +
  geom_histogram(binwidth = 20) +
  coord_cartesian(xlim = c(0, 3000)) +
  labs(x = "RT (ms)", title = "RTs (0-3000 ms)")
# words are much less accurate than nonwords (0.76 vs .95)
# the BLP contains almost all mono- and disyllabic English words, including many very rare ones that participants don't know
# --> they answer "nonword" --> error
# accuracy strongly depends on frequency (freq = occurrences per million words)
df_blp |>
  filter(lex == "word") |>
  mutate(log10_freq = cut(log10(freq), c(-Inf, -1, 0, 1, Inf))) |>
  group_by(log10_freq) |>
  summarise(n = n(), accuracy = mean(acc))
# words below 0.1 per million: about 50% correct (chance level), above 10 per million: about 97%

# clean data

# the EZ uses the variance of RTs, which is very sensitive to outliers
# one 45-second trial can dominate a cell's variance
# also very fast responses (< 200 ms) are guesses rather than decisions
# --> trim and convert RT to seconds
rt_min <- 0.2   # seconds
rt_max <- 2.5   # seconds

df <- df_blp |>
  mutate(rt = rt / 1000) |>
  filter(rt > rt_min, rt < rt_max)

# check how much we removed
1 - nrow(df) / nrow(df_blp)


# defining conditions

# the EZ works with summary statistics per participant x condition, so
# participant-level predictors (e.g., age, vocabulary score) can be continuous
# --> one value per participant --> they can predict that participant's v, a or t0
# and trial-level (item) predictors such as word frequency change from trial to trial
# and are averaged away in the summary statistics --> they must be binned into conditions
# a trial-level DDM could use frequency as a continuous predictor --> one of the costs of the EZ

# frequency on the log10 scale (as done in the paper we took this dataset from)
# cutoff at log10(freq) = 0, i.e. 1 occurrence per million words
# below = low-frequency (LF)
# at or above = high-frequency (HF)
df <- df |>
  mutate(cond = case_when(
    lex == "non-word"  ~ "nonword",
    log10(freq) <  0   ~ "word_lf",
    log10(freq) >= 0   ~ "word_hf"
  ),
  cond = factor(cond, levels = c("nonword", "word_lf", "word_hf")))

df |> count(cond)


# computing the EZ summary statistics 
# per participant x condition ("cell") we need:
# n_trials: number of trials
# n_correct: number of correct responses
# mean_rt: mean RT of the CORRECT responses
# var_rt: variance of RT of the CORRECT responses

# note --> summarise() works top to bottom so if you write `acc = mean(acc)` first
# the later lines would use the mean instead of the trial-level 0/1 values
# --> accuracy is computed last
ez_data <- df |>
  group_by(subj, cond) |>
  summarise(
    n_trials  = n(),
    n_correct = sum(acc),
    mean_rt   = mean(rt[acc == 1]),
    var_rt    = var(rt[acc == 1]),
    accuracy  = mean(acc),
    .groups   = "drop"
  )

ez_data
# 20 participants x 3 conditions = 60 cells: the whole data set is now 60 rows

# what does the frequency effect look like in the summary statistics?
ez_data |>
  pivot_longer(c(accuracy, mean_rt, var_rt), names_to = "statistic") |>
  ggplot(aes(cond, value, group = subj)) +
  geom_line(alpha = 0.3) +
  geom_point(alpha = 0.5) +
  facet_wrap(~ statistic, scales = "free_y") +
  labs(x = NULL, y = NULL, title = "EZ summary statistics per participant")
# LF words: lower accuracy and slower/more variable RTs -> lower drift?


# design matrix for the drift rate

# like in lme4/brms, condition effects enter through a design matrix X
# lex:  dummy coded, 0 = nonword, 1 = word
# freq: -1/2 = LF, +1/2 = HF
# frequency only exists for words --> it enters as an interaction lex:freq
# (for nonwords lex = 0, so their freq value does not matter)

#   cond      Intercept   lex   lex:freq
#   nonword       1        0       0
#   word_lf       1        1     -1/2
#   word_hf       1        1      1/2

# in this coding:
# Intercept = drift for nonwords
# lex       = mean drift for words (average of LF and HF) - drift for nonwords
# lex:freq  = drift for HF words - drift for LF words (frequency effect)
contrasts_cond <- tibble(
  cond = factor(c("nonword", "word_lf", "word_hf"), levels = levels(df$cond)),
  lex  = c(0, 1, 1),
  freq = c(0, -1/2, 1/2)
)

ez_data <- ez_data |> left_join(contrasts_cond, by = "cond")

X <- model.matrix(~ 1 + lex + lex:freq, data = ez_data)
head(X)

# the same coding once per condition (used later to compute condition means)
X_cond <- model.matrix(~ 1 + lex + lex:freq, data = contrasts_cond)
X_cond



# data for Stan 
# Stan cannot handle factors or arbitrary IDs: participants must be numbered 1, 2, ..., J without gaps
# the original IDs in `subj` may have gaps (or be text) so we recode them
# factor() sorts the unique IDs, as.integer() turns them into 1..J
# we keep `subj` in ez_data so we can map results back later
ez_data <- ez_data |> mutate(subj_idx = as.integer(factor(subj)))

ez_data |> distinct(subj, subj_idx)   # mapping old ID -> new index

# Stan takes its data as a named list
# each name must match a variable in the `data { }` block of the model 
# same name, same type, same length
# otherwise Stan stops with an error before sampling
# Each row of ez_data is one cell (participant x condition)
# so all vectors below have length N_cells and are in the same row order
stan_data <- list(
  # tells Stan how long the vectors/matrices are
  N_cells = nrow(ez_data), # number of cells (here 20 x 3 = 60)
  J = max(ez_data$subj_idx), # number of participants (20)
  K = ncol(X), # number of drift coefficients (3 --> intercept, lex, lex:freq)
  
  # which cell belongs to whom, and to which condition
  subj = ez_data$subj_idx,# participant index (1..J) of each cell
  # used to pick that person's a, t0 and random effects
  X = X, # design matrix (N_cells x K): the condition coding of each cell
  
  # data: the EZ summary statistics of each cell
  n_trials   = ez_data$n_trials,       # trials in the cell
  n_correct  = ez_data$n_correct,      # correct responses in the cell
  mean_rt    = ez_data$mean_rt,        # mean correct RT (seconds)
  var_rt     = ez_data$var_rt,         # variance of correct RTs (seconds^2)
  
  # fit the model to the data (0) or sample from the priors only (1)
  # 0 = normal fit: priors + likelihood -> posterior
  # 1 = the likelihood is switched off (check `if (prior_only == 0)` in the model block)
  # --> so Stan samples only from the priors
  # we use this to see what the priors predict before seeing the data
  prior_only = 0
)

# correct names, lengths and types? (int vs num, no NAs)
str(stan_data)


# now we can compile the model 
# lets look at the stan file 
ez_model <- cmdstan_model(file.path("hier_ez_blp.stan"))

# prior predictive check 
# before looking at the data: what do our priors imply about accuracy and RTs?
# same model -->  prior_only = 1 switches off the likelihood 
fit_prior <- ez_model$sample(
  data            = modifyList(stan_data, list(prior_only = 1)),
  chains          = 4,
  parallel_chains = 4,
  iter_warmup     = 500,
  iter_sampling   = 500,
  seed            = 1,
  refresh         = 0,
  output_dir      = "fits",
  output_basename = "ez_blp_prior"
)
# this prior fit shows a few divergent transitions (about 10 of 2000 draws)
# they only appear when sampling from the priors alone, not in the real fit
# for the prior predictive plot this does not matter, we only look at the rough range of predicted data

prior_pred <- fit_prior$draws(c("pred_n_correct", "pred_mean_rt"),
                              format = "draws_matrix")
prior_acc  <- prior_pred[, grepl("pred_n_correct", colnames(prior_pred))] /
  matrix(stan_data$n_trials, nrow(prior_pred), stan_data$N_cells,
         byrow = TRUE)
prior_mrt  <- prior_pred[, grepl("pred_mean_rt", colnames(prior_pred))]

tibble(accuracy = as.vector(prior_acc), mean_rt = as.vector(prior_mrt)) |>
  slice_sample(n = 5000) |>
  ggplot(aes(mean_rt, accuracy)) +
  geom_point(alpha = 0.1) +
  geom_point(data = ez_data, colour = "red") +
  coord_cartesian(xlim = c(0, 3)) +
  labs(x = "mean RT (s)", title = "Prior predictions (grey) vs observed cells (red)")
# the priors should cover the observed data without being absurdly wide

# each grey point = one cell (participant x condition) simulated from one prior draw
# each red point = one observed cell

# these priors are good (weakly informative):
# all observed cells (red) lie inside the grey cloud: the priors can
# produce data like ours, so they do not rule out the true values
# most prior mass is in a plausible region but the cloud is much wider than the data
# the priors do not decide the result --> the data will
# no mean RTs near 0: the prior on t0 (median ~350 ms) makes RTs below ~200 ms very rare

# notes:
# many predictions are at accuracy ~1: large a and large v together make errors almost impossible
# This is plausible for easy items, but a bit optimistic in general

# some predictions are below .5 (below chance): the drift prior allows negative drift
# this looks strange, but its important: for LF words some participants are at or below .5,
# because they do not know many of these words and answer "nonword"
# A prior that forced v > 0 would clash with these cells
# --> a thin tail goes to mean RTs of 2-3 s: so unlikely for lexical decision,
# but rare in the prior and the data will easily overrule it

# this plot only shows accuracy and mean RT
# the priors also imply RT variances, which we could check the same way

# with 20 participants x ~200-600 trials per cell, the likelihood dominates these priors
# we can later change the priors and see whether the posterior moves


# fitting the model 
fit <- ez_model$sample(
  data            = stan_data,  
  chains          = 4,          # 4 independent chains, each with different random starting values
  # --> if they all end up in the same place, we can trust the result
  parallel_chains = 4,          # run the chains at the same time (one per CPU core)
  iter_warmup     = 1000,       # warmup: the sampler tunes itself (step size etc.)
  # --> these draws are thrown away
  iter_sampling   = 1000,       # draws kept per chain --> 4 x 1000 = 4000 posterior draws
  adapt_delta     = 0.95,       # target acceptance rate (default 0.8): higher = smaller steps,
  # slower but fewer divergences in hierarchical models
  seed            = 2026,       # same seed = same results every time (reproducible)
  refresh         = 250,        # print progress every 250 iterations
  output_dir      = "fits",     # save the draws (CSV files) in the fits folder 
  output_basename = "ez_blp"    
)
# nice about the EZ: it only needs 60 cells instead of 24,000 trials so its quite fast
# you may see "Rejecting initial value" or "Informational Message" at the start
# This is because Stan tried random starting values that are impossible for the model and picks new ones
# --> harmless if it only happens at the start
# the diagnostics below look fine


# checking if the sampler worked

# divergences/tree depth --> ideally all 0 
# ebfmi: one value per chain, should be > 0.3 (Stan warns below that)
# --> low values = sampler explores the posterior poorly
fit$diagnostic_summary()

# Rhat close to 1 (< 1.01) and large ESS for the parameters
main_pars <- c("beta_v", "sd_v", "mu_log_a", "sd_log_a", "mu_log_t0", "sd_log_t0")
fit$summary(main_pars)

# trace plots: chains should look like "hairy caterpillars" 
mcmc_trace(fit$draws(c("beta_v", "mu_log_a", "mu_log_t0")))


# checking the results at population level 

# population-level drift coefficients 
# one row per column of X: mean, sd and 95% credible interval of the posterior
fit$summary("beta_v", mean, sd, ~ quantile(.x, probs = c(0.025, 0.975))) |>
  mutate(term = colnames(X))
# intercept = drift for nonwords (about 2.0): clearly above 0 --> people accumulate
# evidence toward the correct response
# lex = mean drift for words - drift for nonwords (about -0.55) --> negative: words have on average lower drift than nonwords
# that is surprising at first, but the word average includes the LF words, which are very hard
# lex:freq = how much higher the drift is for HF words than for LF words --> this is the word frequency effect
# --> speed of evidence accumulation --> the interval is far from 0
# we later turn these coefficients back into one drift per condition

# all posterior draws as a data frame: one row per draw (4 chains x 1000 = 4000 rows)
# one column per parameter
draws <- as_draws_df(fit$draws(c("beta_v", "mu_log_a", "mu_log_t0",
                                 "sd_v", "sd_log_a", "sd_log_t0")))

# frequency effect: how much faster do people accumulate evidence for HF than for LF words?
# how sure are we that it is positive?
mcmc_areas(fit$draws("beta_v[3]"), prob = 0.95) +
  labs(title = "Frequency effect on drift (HF - LF)")
mean(draws$`beta_v[3]` > 0)    # posterior probability that the effect is > 0
# = proportion of draws above 0
# here 1: all 4000 draws are > 0
# report as P > .999 (with 4000 draws we cannot resolve smaller probabilities)
# this probability is conditional on the model and the priors


# average drift per condition: combine coefficients with the contrasts
# the coefficients (intercept, lex, lex:freq) are hard to read directly, so we turn
# them back into one drift per condition: drift = X_cond row x beta_v
# e.g. LF words: 1 * intercept + 1 * lex + (-1/2) * lex:freq
# we do this for every posterior draw, so we also estimate teh uncertainty
# always compute first, then summarise (do not only the posterior means)
beta_mat <- fit$draws("beta_v", format = "draws_matrix")   # draws x K (4000 x 3)
v_cond   <- beta_mat %*% t(X_cond) # matrix multiplication -> draws x conditions (4000 x 3)
colnames(v_cond) <- as.character(contrasts_cond$cond)

# summarise the 4000 draws per condition: posterior mean and 95% credible interval
v_cond_summary <- as_tibble(v_cond) |>
  pivot_longer(everything(), names_to = "cond", values_to = "v") |>   # long format: one row per draw x condition
  mutate(cond = factor(cond, levels = levels(df$cond))) |>            # keep the order nonword, LF, HF
  group_by(cond) |>
  summarise(mean  = mean(v),
            lower = quantile(v, 0.025),
            upper = quantile(v, 0.975))
v_cond_summary
# what we see:
# LF words have a much lower drift: evidence for "word" builds up slowly
# which is because of slower and less accurate responses
# HF words and nonwords are similar: both are "easy" decisions
# this also explains the negative lex coefficient: the word average is below the nonword drift 

ggplot(v_cond_summary, aes(mean, cond)) +
  geom_pointrange(aes(xmin = lower, xmax = upper)) +
  labs(x = "population drift rate (95% CrI)", y = NULL)


# boundary and non-decision time: back-transform from the log scale
# in the model, log(a) and log(t0) are normally distributed across participants
# exp(mu) is the population median of a and t0 (not the mean)
draws |>
  mutate_variables(a_median     = exp(mu_log_a),            # new variables, computed per draw
                   t0_median_ms = exp(mu_log_t0) * 1000) |>  # t0 in ms
  subset_draws(variable = c("a_median", "t0_median_ms")) |>
  summarise_draws(mean, ~ quantile(.x, probs = c(0.025, 0.975)))
# what we see:
# boundary separation (response caution) is the same for all conditions by design
# NDT is at 360 ms, this is the time used for encoding the letter string & pressing the key


# how much do people differ? (by-participant SDs)
# note: fit$summary() shows 90% intervals by default
fit$summary(c("sd_v", "sd_log_a", "sd_log_t0"))
# what we see:
# sd_v[1]: participants differ in their nonword drift by about +/- 0.5 around 2.0
# sd_v[2], sd_v[3]: the lex and freq effects also differ between people (like by-participant random slopes in lme4)
# sd_log_a and sd_log_t0 are on the log scale: roughly, people
# differ by about 16% in caution and 12% (~ +/- 40 ms) in non-decision time


# participant level 

# drift per participant and condition (the cell-level v from the model)
# v has one value per row of ez_data (= per cell), in the same order
# --> we can bind the participant and condition labels back on
v_cells <- fit$summary("v", mean, ~ quantile(.x, probs = c(0.025, 0.975))) |>
  bind_cols(ez_data |> select(subj, cond))

ggplot(v_cells, aes(cond, mean, group = subj)) +
  geom_line(alpha = 0.4) +
  geom_point() +
  labs(x = NULL, y = "drift rate (posterior mean)",
       title = "Each line is one participant")
# what we see:
# every participant shows the frequency effect (drop at LF words)
# but the size differs: for some, LF drift is near 0 or even slightly
# negative (they respond "nonword" to many LF words, i.e. accuracy <= .5)
# these are partially pooled estimates: participants with noisy data are
# pulled toward the group 


# all parameters per participant: a, t0 and the three drift coefficients
# fit$summary returns names like "a[3]", "t0[12]" or "b_v[5,2]"
# (b_v[5,2] = participant 5, drift coefficient 2 = lex)
# we extract the participant index and the parameter name
drift_labels <- c("drift: nonword", "drift: word - nonword", "drift: HF - LF")

person_pars <- fit$summary(c("a", "t0", "b_v"), mean,
                           ~ quantile(.x, probs = c(0.025, 0.975))) |>
  mutate(
    subj_idx = as.integer(sub("^.*\\[(\\d+).*$", "\\1", variable)),   # "b_v[5,2]" -> 5
    k        = as.integer(ifelse(grepl(",", variable),
                                 sub("^.*,(\\d+)\\]$", "\\1", variable), NA)),  # "b_v[5,2]" -> 2
    par      = case_when(
      startsWith(variable, "a[")  ~ "a (boundary)",
      startsWith(variable, "t0[") ~ "t0 (s)",
      TRUE                        ~ drift_labels[k]
    ),
    par = factor(par, levels = c("a (boundary)", "t0 (s)", drift_labels))
  )

# sort participants by their frequency effect, same order in every panel
subj_order <- person_pars |>
  filter(par == "drift: HF - LF") |>
  arrange(mean) |>
  pull(subj_idx)

ggplot(person_pars, aes(mean, factor(subj_idx, levels = subj_order))) +
  # dashed line at 0 for the two effects
  geom_vline(data = tibble(par = factor(drift_labels[2:3], levels = levels(person_pars$par))),
             aes(xintercept = 0), linetype = "dashed", colour = "grey50") +
  geom_pointrange(aes(xmin = `2.5%`, xmax = `97.5%`), size = 0.2) +
  facet_wrap(~ par, scales = "free_x", nrow = 1) +
  labs(x = "posterior mean (95% CrI)", y = "participant (sorted by frequency effect)")
# what we see:
# each row is one participant
# the frequency effect is positive for every participant (all intervals far from 0) but its size differs
# the lexicality effect differs in sign: most participants have lower drift for
# words than for nonwords, a few have higher drift for words
# the intervals are narrow because each participant has about 1200 trials


# comparison to the non-hierarchical EZ 
# the non-hierarchical ez computes v, a, t0 for each cell in closed form, separately,
# with no pooling across participants
ez_non_hierarchical <- function(Pc, VRT, MRT, n, s = 1) {
  Pc <- ifelse(Pc == 1, 1 - 1 / (2 * n), Pc)   # edge correction
  L  <- qlogis(Pc)
  x  <- L * (L * Pc^2 - L * Pc + Pc - 0.5) / VRT
  v  <- sign(Pc - 0.5) * s * x^(1/4)
  a  <- s^2 * L / v
  y  <- -v * a / s^2
  MDT <- (a / (2 * v)) * (1 - exp(y)) / (1 + exp(y))
  tibble(v_non_hierarchical = v, a_non_hierarchical = a, t0_non_hierarchical = MRT - MDT)
}

compare_v <- ez_data |>
  bind_cols(ez_non_hierarchical(ez_data$accuracy, ez_data$var_rt,
                                ez_data$mean_rt, ez_data$n_trials)) |>
  mutate(v_hier = v_cells$mean)

ggplot(compare_v, aes(v_non_hierarchical, v_hier, colour = cond)) +
  geom_abline(linetype = "dashed") +
  geom_point(size = 2) +
  labs(x = "non-hierarchical EZ (per cell, no pooling)",
       y = "hierarchical EZ (posterior mean)",
       title = "Drift: non-hierarchical vs hierarchical EZ")
# we can see partial pooling: the hierarchical model borrows strength across
# participants and pulls extreme cells toward the group

# the non-hierarchical EZ estimates a and t0 separately for every
# cell, our model holds them constant within a participant
# a condition difference must then go through the drift rate

# the non-hierarchical EZ also breaks down when accuracy is exactly .5 (v = 0, a undefined)
# and gives no uncertainty
# the hierarchical model gives a full posterior for every cell


# posterior predictive check
# does the fitted model reproduce the summary statistics it was fitted to?
# for every posterior draw, the model simulates new data for every cell
# (pred_n_correct, pred_mean_rt, pred_var_rt in the generated quantities block)
# these are 4000 simulated data sets, if the model describes the data well, the
# observed values should look like "typical simulated" values
# this means the intervals include both, the uncertainty about the parameters and the
# random noise of new data (sampling noise)

# one table with three parts (accuracy, mean RT, RT variance),
# each with one row per cell: predicted mean, 90% interval, observed value
ppc <- bind_rows(
  # accuracy: the model predicts numbers of correct responses
  # we divide by n_trials to get proportions, comparable to the observed accuracy
  fit$summary("pred_n_correct", mean, ~ quantile(.x, probs = c(0.05, 0.95))) |>
    mutate(statistic = "accuracy",
           across(c(mean, `5%`, `95%`), \(x) x / ez_data$n_trials),
           observed = ez_data$accuracy),
  # mean RT of correct responses (seconds)
  fit$summary("pred_mean_rt", mean, ~ quantile(.x, probs = c(0.05, 0.95))) |>
    mutate(statistic = "mean_rt", observed = ez_data$mean_rt),
  # variance of correct RTs (seconds^2)
  fit$summary("pred_var_rt", mean, ~ quantile(.x, probs = c(0.05, 0.95))) |>
    mutate(statistic = "var_rt", observed = ez_data$var_rt)
) |>
  # all three parts are in the row order of ez_data 
  # so we repeat the condition labels 3 times
  mutate(cond = rep(ez_data$cond, 3))


# x = observed, y = predicted, one point per cell (participant x condition)
# dashed line = perfect prediction
# a point whose interval crosses the dashed line: the model can produce
# the observed value
# a point above the line = over-predicted,
# below = under-predicted
ggplot(ppc, aes(observed, mean, colour = cond)) +
  geom_abline(linetype = "dashed") +
  geom_pointrange(aes(ymin = `5%`, ymax = `95%`)) +
  facet_wrap(~ statistic, scales = "free") +
  labs(x = "observed", y = "predicted (90% interval)")
# we check the model on the same data we fitted it to
# this shows whether the model can reproduce the data and not how well it predicts new data


# this is the share of cells whose observed value lies inside the 90% predictive interval
# and the average misfit (observed - predicted), per condition
ppc |>
  group_by(statistic, cond) |>
  summarise(coverage_90 = mean(observed >= `5%` & observed <= `95%`),
            mean_misfit = mean(observed - mean),
            .groups = "drop")
# is the misfit random, or systematic per condition?
# Here it is systematic: nonword accuracy is under-predicted, HF-word accuracy
# is over-predicted, and LF-word RT variance is over-predicted
# one drift rate per condition, with a and t0 shared across conditions, cannot
# produce very accurate but not-faster nonword responses at the same time as
# fast, less accurate word responses
# A typical explanation is a response bias (starting point closer to the "nonword" boundary)
# the EZ assumes an unbiased starting point, so this misfit is a limitation of EZ
# and a reason to consider the EZ with bias


# possible excercises
# change rt_max to 2s or 4s --> Which parameters move and why?

# split words into three bins (tertiles) instead of two, adapt contrasts_cond / X, the Stan model does not need to change

# let a (or t0) differ between words and nonwords
# what would you change in the Stan model? does the posterior predictive check improve?
# is it psychologically plausible?

# use the full BLP data (`df_blp_complete` with 78 participants)
# do the population estimates change? do the intervals get narrower?

# change the priors in the model in the Stan file and re-run the
# prior predictive check
# how sensitive are the results?