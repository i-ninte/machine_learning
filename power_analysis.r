# Load necessary library for power analysis
library(pwr)

# Set parameters
alpha <- 0.05          # Significance level
power <- 0.80          # Desired power level
effect_size <- 0.5     # Effect size (Cohen's d)
sample_size <- NULL    # We want to find the required sample size

# Calculate sample size for a two-sample t-test
sample_size <- pwr.t.test(d = effect_size, sig.level = alpha, power = power)$n

# Print the required sample size
cat("Required Sample Size:", sample_size, "\n")
