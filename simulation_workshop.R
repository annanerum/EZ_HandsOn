library(rtdists)
library(dplyr)
PROJECT_DIR <- "/Users/annegiacobello/Desktop/EZ Paper /workshop utrecht"
DATA_DIR    <- file.path(PROJECT_DIR, "data_sim")


# condition1: +1 = nonword, -1 = word (balanced)
# condition2: centred log word frequency, 0 for nonwords

# discrete levels, so frequency is constant within an EZ cell
# 4 EZ cells per subject: nonword, word_HF, word_LF, word_VLF
# word_HF: high-frequency words 
# word_LF: low-frequency words
# word_VLF: very-low-frequency words 
# I took this from another study and changed it a little (Ratcliff, Gomez & McKoon, 2004)


# drift: intercept + condition1 + frequency
# alpha and tau intercept-only
# no starting point bias (w = 0.5), no across-trial variability so data match EZ assumptions
# hierarchical: subject-level random effects on nu_intercept, nu_c, nu_c2, alpha_intercept, tau_intercept_raw (w has SD 0)


n_subj   <- 100
n_trials <- 300
# 300 (150 nonwords + 50 words per frequency level).
# more trials per cell = less noisy EZ summary stats
# must be divisible by 6.

# frequency levels + fixed item list 
# replaces rnorm(mean = 17, sd = 3.5) per trial bc EZ only sees cell summaries
# so frequency must be constant within a cell (otherwise the spread in drift acts like sv whcih we dont model)
# and centring keeps beta_nu1 / nu_c interpretable 
freq_levels <- c(HF = 1, LF = 0, VLF = -1)   

n_words     <- n_trials / 2
n_per_level <- n_words / length(freq_levels)
stopifnot(n_per_level == round(n_per_level))

# frequency is an item property -> same items (and frequencies) for every subject, only the trial order is shuffled per person
item_list <- data.frame(
  item       = seq_len(n_trials),
  cell       = c(rep("nonword", n_words),
                 rep(paste0("word_", names(freq_levels)), each = n_per_level)),
  condition1 = rep(c(1, -1), each = n_words),
  condition2 = c(rep(0, n_words), rep(unname(freq_levels), each = n_per_level))
)


true_pop <- list(
  beta_nu1    = 2.50,       
  nu_c        = -0.25,      
  # nonword (+1):              2.50 - 0.25 = 2.25
  # word (-1), avg frequency:  2.50 + 0.25 = 2.75
  # words easier than nonwords
  nu_c2       = 0.80,       
  beta_alpha1 = log(1.2),   
  beta_tau    = 0.40,       
  w           = 0.5,        
  sw          = 0,          
  sv          = 0,          
  st0_raw     = 0           
)
# new values are similar to the lexical descision task from Ratcliff, Gomez & McKoon (2004) 


sd_subj <- list(
  sigma_nu_intercept    = 0.40, 
  sigma_nu_c1           = 0.15, 
  sigma_nu_c2           = 0.15, 
  sigma_alpha_intercept = 0.08, 
  sigma_tau_intercept   = 0.05, 
  sigma_w               = 0     
)


# sanity check: predicted accuracy and mean RT per cell at population values
ez_forward <- function(v, a, t0) {
  y <- exp(-a * v)
  data.frame(v   = v,
             pc  = round(1 / (1 + y), 3),
             mrt = round(t0 + (a / (2 * v)) * (1 - y) / (1 + y), 3))
}
pop_v <- c(nonword = true_pop$beta_nu1 + true_pop$nu_c,
           setNames(true_pop$beta_nu1 - true_pop$nu_c + true_pop$nu_c2 * freq_levels,
                    paste0("word_", names(freq_levels))))
print(ez_forward(pop_v, a = exp(true_pop$beta_alpha1), t0 = true_pop$beta_tau))


simulate_subject <- function(sp) {
  # sp = one row of subj_par --> one persons parameters
  tr <- item_list[sample(n_trials), ]
  # trial list = fixed item list (150 nonwords + 50 words), shuffled per person
  condition1 <- tr$condition1
  condition2 <- tr$condition2
  # condition2 = item's centred frequency level instead of a new rnorm draw per trial
  
  a_subj  <- exp(sp$alpha_intercept)  # constant per person across trials (varies by person)
  # boundary separation
  # intercept-only, no trial-to-trial variation and no condition effect
  
  v_trial <- sp$nu_intercept + sp$nu_c * condition1 + sp$nu_c2 * condition2  # person's drift rate per trial
  # person-specific intercept, condition effect, frequency effect (words only, freq = 0 for nonwords)
  
  z_abs  <- sp$w * a_subj  # person specific starting point
  # converts relative (0-1) to real
  
  sz_abs <- true_pop$sw * a_subj
  # trial-to-trial starting point variability, also converted
  
  sim <- rdiffusion(
    n = n_trials, a = a_subj, v = v_trial, t0 = sp$t0,
    z = z_abs, sz = sz_abs, sv = true_pop$sv, st0 = sp$st0
  )
  # drawing n_trials random RT and response pairs from the diffusion model
  
  data.frame(
    subj       = sp$subj,             
    trial      = seq_len(n_trials),   # trial within subject
    item       = tr$item,             # item id (same items for everyone)
    cell       = tr$cell,             # which EZ cell 
    condition1 = condition1,          # +1 = nonword, -1 = word
    condition2 = condition2,          # centred log frequency (0 for nonwords)
    a          = a_subj,              # this ppns boundary, repeated every row
    v          = v_trial,             # this trials drift rate, one value per cell per ppn
    rt         = sim$rt,              # simulated reaction time in seconds
    acc        = ifelse(sim$response == "upper", 1L, -1L) # +1 = correct/upper, -1 = error/lower
  )
}



run_one_rep <- function(rep_id, seed) {
  set.seed(seed)  
  
  z_nu_c  <- rnorm(n_subj)
  z_alpha <- rnorm(n_subj)
  z_tau   <- rnorm(n_subj)
  z_w     <- rnorm(n_subj)
  z_nu1   <- rnorm(n_subj)
  z_nu_c2 <- rnorm(n_subj)
  # z_nu_c2 moved to the end
  # inserted after z_nu_c shifted all later draws 
  
  subj_par <- data.frame(
    subj              = 1:n_subj,
    nu_c              = true_pop$nu_c        + sd_subj$sigma_nu_c1           * z_nu_c,
    # each ppns condition effect on drift
    nu_c2             = true_pop$nu_c2       + sd_subj$sigma_nu_c2           * z_nu_c2,
    # each ppns frequency effect on drift
    nu_intercept      = true_pop$beta_nu1    + sd_subj$sigma_nu_intercept    * z_nu1,
    # each ppns drift intercept
    alpha_intercept   = true_pop$beta_alpha1 + sd_subj$sigma_alpha_intercept * z_alpha,
    # each ppns log boundary sep intercept
    tau_intercept_raw = true_pop$beta_tau    + sd_subj$sigma_tau_intercept   * z_tau,
    # each ppns t0 in seconds
    w                 = plogis(qlogis(true_pop$w) + sd_subj$sigma_w * z_w)
    # each ppns starting point --> kept in (0,1) by logit transform
  )
  
  subj_par$t0  <- subj_par$tau_intercept_raw
  # tau_intercept_raw is already on the natural scale (seconds) --> used directly as t0
  stopifnot(all(subj_par$t0 > 0))
  
  subj_par$st0 <- true_pop$st0_raw * 2 * subj_par$t0
  # each subject's trial-to-trial NDT variability, scaled relative to their own t0
  
  # true parameters per subject x cell (what EZ estimates per cell) -> for recovery
  cells <- unique(item_list[, c("cell", "condition1", "condition2")])
  true_cells <- merge(subj_par, cells, by = NULL)
  true_cells$v  <- with(true_cells, nu_intercept + nu_c * condition1 + nu_c2 * condition2)
  true_cells$a  <- exp(true_cells$alpha_intercept)
  true_cells <- true_cells[order(true_cells$subj, true_cells$cell),
                           c("subj", "cell", "condition1", "condition2", "v", "a", "t0")]
  stopifnot(all(true_cells$v > 0))  # no negative drifts (would mean below-chance accuracy)
  
  sim_list <- lapply(seq_len(n_subj), function(i) simulate_subject(subj_par[i, ]))
  
  sim_data <- do.call(rbind, sim_list)
  
  rownames(sim_data) <- NULL
  stopifnot(all(table(sim_data$subj) == n_trials))
  
  min_rt_obs <- tapply(sim_data$rt, sim_data$subj, min)
  gap_min_rt_t0 <- min_rt_obs[as.character(subj_par$subj)] - subj_par$t0
  # how far each person's fastest RT lies above their true t0
  
  # EZ summary statistics per subject x cell (input for the model fit)
  ez_summary <- sim_data %>%
    group_by(subj, cell, condition1, condition2) %>%
    summarise(N               = n(),
              n_correct       = sum(acc == 1),
              mean_rt_correct = mean(rt[acc == 1]),
              var_rt_correct  = var(rt[acc == 1]),
              .groups = "drop")
  
  out_dir <- file.path(DATA_DIR, sprintf("rep%02d", rep_id))
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  write.csv(sim_data, file.path(out_dir, "data_sim.csv"), row.names = FALSE)
  # trial-level data 
  write.csv(ez_summary, file.path(out_dir, "ez_summary.csv"), row.names = FALSE)
  # cell-level summary stats for fitting ez
  saveRDS(subj_par,      file.path(out_dir, "true_subject_parameters.rds"))
  # true person-level parameters
  saveRDS(true_cells,    file.path(out_dir, "true_cell_parameters.rds"))
  # true v, a and t0 per person x cell
  saveRDS(true_pop,      file.path(out_dir, "true_population_parameters.rds"))
  # true population-level parameters
  saveRDS(min_rt_obs,    file.path(out_dir, "min_rt_observed.rds"))
  # per person observed minimum RT
  
  cat(sprintf("rep%02d done | seed=%d | mean_rt=%.3f | mean_t0=%.3f | acc=%.3f | min_gap=%.3f | perfect_cells=%d/%d\n",
              rep_id, seed, mean(sim_data$rt), mean(subj_par$t0),
              mean(sim_data$acc == 1), min(gap_min_rt_t0),
              sum(ez_summary$n_correct == ez_summary$N), nrow(ez_summary)))
  # prints how many subject x cells have 100% accuracy (EZ ceiling problem)
  invisible(sim_data)
}

n_reps    <- 10   # how many simulated datasets
base_seed <- 8181 # starting seed
for (r in seq_len(n_reps)) run_one_rep(rep_id = r, seed = base_seed + r)


#### plotting

# all reps into one data frame
dat_all <- bind_rows(lapply(seq_len(n_reps), function(r)
  read.csv(file.path(DATA_DIR, sprintf("rep%02d", r), "data_sim.csv")) %>%
    mutate(rep = r)))

# cell effects per rep
cond_tab <- dat_all %>%
  group_by(rep, cell) %>%
  summarise(mean_rt = mean(rt), median_rt = median(rt), acc = mean(acc == 1), .groups = "drop")
print(cond_tab, n = 40)
# grouped by cell instead of condition2 

# cell effects summarised over reps (mean, min, max)
cond_tab %>%
  group_by(cell) %>%
  summarise(across(c(mean_rt, median_rt, acc), list(mean = mean, min = min, max = max)))
# removed the cond_tab2 block 

# per-person summaries
subj_all <- dat_all %>%
  group_by(rep, subj) %>%
  summarise(rt_effect_cond = mean(rt[condition1 == -1]) - mean(rt[condition1 == 1]),
            # word - nonword (expected < 0 now)
            freq_effect    = mean(rt[cell == "word_VLF"]) - mean(rt[cell == "word_HF"]),
            acc_nonword    = mean(acc[cell == "nonword"]  == 1),
            acc_HF         = mean(acc[cell == "word_HF"]  == 1),
            acc_LF         = mean(acc[cell == "word_LF"]  == 1),
            acc_VLF        = mean(acc[cell == "word_VLF"] == 1),
            acc_all        = mean(acc == 1),
            .groups = "drop")
# accuracy per cell instead of per condition1

# per rep: reversed effects, ceiling, lowest accuracies
subj_all %>%
  group_by(rep) %>%
  summarise(share_cond_reversed = mean(rt_effect_cond > 0),
            share_freq_reversed = mean(freq_effect < 0),
            share_HF_perfect    = mean(acc_HF == 1),     # ceiling check
            min_acc_VLF         = min(acc_VLF),
            min_acc_nonword     = min(acc_nonword),
            min_acc_all         = min(acc_all))


# pooled plots over all reps
par(mfrow = c(2, 3))
hist(subj_all$rt_effect_cond, breaks = 40, main = "RT word - nonword per person", xlab = "RT(word) - RT(nonword) in s")
abline(v = 0, lty = 2)
hist(subj_all$freq_effect, breaks = 40, main = "RT frequency effect per person", xlab = "RT(VLF) - RT(HF) in s")
abline(v = 0, lty = 2)
hist(dat_all$rt, breaks = 100, main = "RT distribution (all trials, all reps)", xlab = "RT in s")
hist(subj_all$acc_nonword, breaks = 30, main = "Accuracy (nonwords)", xlab = "proportion correct", xlim = c(0, 1))
hist(subj_all$acc_HF, breaks = 30, main = "Accuracy (HF words)", xlab = "proportion correct", xlim = c(0, 1))
hist(subj_all$acc_VLF, breaks = 30, main = "Accuracy (VLF words)", xlab = "proportion correct", xlim = c(0, 1))
abline(v = 0.5, lty = 2)
par(mfrow = c(1, 1))