# Load necessary libraries
library(tidyverse)
library(bartMachine)
library(tictoc)
library(feather)
library(groupdata2)
library(terra)
library(caret)


# Specify Paths
input_path <- "data/"  # Adjust path as needed
output_path <- "output/"

# Load Combined Data
admin2_data <- read.csv(paste0(input_path, "covariates/district/all_features_districts.csv"))

# Define response variable as log of population density
admin2_data <- admin2_data %>%
  mutate(pop_density = log(T_TL / district_area)) %>%
  filter(!is.na(pop_density))

# Select covariates (exclude identifiers and response-related columns)
covs_admin2 <- admin2_data %>%
  select(-ADM2_PT, -ADM2_PCODE, -T_TL, -district_area, -pop_density, -log_population)

# Select Covariates - Trees, Built.Area, building_area, building_count, osm_roads, osm_potw
covs_admin2 <- covs_admin2 %>%
  select(Trees, Built.Area, building_area, building_count, osm_roads, osm_pofw, DNB, Water, SR_B6, SR_B1)

# Calculate mean and standard deviation of covariates for scaling
cov_stats <- data.frame(
  Covariate = colnames(covs_admin2),
  Mean = apply(covs_admin2, 2, mean, na.rm = TRUE),
  Std_Dev = apply(covs_admin2, 2, sd, na.rm = TRUE)
)

# Scaling function to standardize covariates
stdize <- function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}

# Apply scaling function
covs_admin2 <- apply(covs_admin2, 2, stdize) %>% as_tibble()

# Combine scaled covariates with response variable
admin2_data_scaled <- admin2_data %>%
  select(ADM2_PT, T_TL, pop_density) %>%
  cbind(covs_admin2)

set.seed(1)
#use 70% of dataset as training set and 30% as test set
smp_size <- floor(0.70 * nrow(admin2_data_scaled))

## set the seed to make your partition reproducible
set.seed(123)
sample <- sample(seq_len(nrow(admin2_data_scaled)), size = smp_size)

train  <- admin2_data_scaled[sample, ]
train_covs <- train %>% select(all_of(colnames(covs_admin2)))
test   <- admin2_data_scaled[-sample, ]
test_covs <- test %>% select(all_of(colnames(covs_admin2)))


# Train the BART Model
set.seed(1234)
model2 <- bartMachine(
  X = train_covs,
  y = train$pop_density,
  k = 5, nu = 10, q = 0.75, num_trees = 700, use_missing_data = TRUE
)
# model2 <- bartMachineCV(
#   X = covs_admin2,
#   y = admin2_data_scaled$pop_density,
#   k_folds = 5
# )

jpeg("output/BART/convergence_diagnostics.jpg", width = 800, height = 600)
# Check for model convergence
plot_convergence_diagnostics(model2)
dev.off()

# Compute credible intervals
admin2_CI <- calc_credible_intervals(model2, new_data = train_covs) %>%
  as_tibble() %>%
  mutate(ci_lower_bd = ci_lower_bd, ci_upper_bd = ci_upper_bd)

# Predictions (On all data)
admin2_predictions <- model2$y_hat_train %>% as_tibble()

# Combine predictions with observed values
admin2_predictions <- admin2_predictions %>%
  cbind(train$pop_density, admin2_CI) %>%
  mutate(
    observed = train$pop_density,
    predicted = value,
    residual = predicted - observed,
    model = "BART"
  )

# Save results
write.csv(admin2_predictions, paste0(output_path, "BART/BART_model_results.csv"))

# Goodness-of-fit metrics
admin2_metrics <- admin2_predictions %>%
  summarise(
    Bias = mean(residual),
    Imprecision = sd(residual),
    Inaccuracy = mean(abs(residual)),
    mse = mean(residual^2),
    rmse = sqrt(mse),
    corr = cor(predicted, observed),
    In_IC = mean(observed < ci_lower_bd & observed > ci_upper_bd) * 100
  )

# Print the metrics
print("Goodness-of-fit metrics:")
print(admin2_metrics)

# Save metrics to a CSV file
write.csv(admin2_metrics, paste0(output_path, "BART/model_metrics.csv"), row.names = FALSE)

# Print the admin2_predictions for verification
print("Admin Level 2 Predictions:")
print(admin2_predictions)

#Predictions on test data
predicted_values <- predict(model2, new_data = test_covs)

# Create a data frame for predictions and observed values
admin2_predictions_test <- data.frame(
  predicted = predicted_values,    
  observed = test$pop_density     
)

# Compute credible intervals
admin2_CI_test <- calc_credible_intervals(model2, new_data = test_covs) %>%
  as_tibble() %>%
  mutate(ci_lower_bd = ci_lower_bd, ci_upper_bd = ci_upper_bd)

# Combine predictions with observed values
admin2_predictions_test <- admin2_predictions_test %>%
  cbind(test$pop_density, admin2_CI_test) %>%
  mutate(
    residual = predicted - observed,
    model = "BART"
  )

# Save results
write.csv(admin2_predictions_test, paste0(output_path, "BART/BART_model_results.csv"))

# Goodness-of-fit metrics
admin2_metrics_test <- admin2_predictions_test %>%
  summarise(
    Bias = mean(residual),
    Imprecision = sd(residual),
    Inaccuracy = mean(abs(residual)),
    mse = mean(residual^2),
    rmse = sqrt(mse),
    corr = cor(predicted, observed),
    In_IC = mean(observed < ci_lower_bd & observed > ci_upper_bd) * 100
  )

# Print the metrics
print("Goodness-of-fit metrics:")
print(admin2_metrics_test)

# Save metrics to a CSV file
write.csv(admin2_metrics_test, paste0(output_path, "BART/model_metrics.csv"), row.names = FALSE)


# Plot Observed vs. Predicted
library(ggplot2)
ggplot(admin2_predictions) +
  geom_pointrange(
    aes(x = observed, y = predicted, ymin = ci_lower_bd, ymax = ci_upper_bd),
    fill = 'grey50', color = 'grey70', shape = 21
  ) +
  geom_abline(slope = 1, intercept = 0, color = 'orange', linewidth = 1) +
  theme_minimal(base_family = "sans", base_size = 14) +
  theme(panel.background = element_rect(fill = "white")) +
  labs(title = '', x = 'Observed Population Density', y = 'Predicted Density')

ggsave("observed_vs_predicted.jpg", plot = last_plot(), path = "output/BART/")

# Plot Residual Histogram
admin2_predictions %>%
  ggplot(aes(residual)) +
  geom_histogram(fill = 'blue', bins = 30, alpha = 0.7) +
  theme_minimal(base_family = "sans", base_size = 14) +
  theme(panel.background = element_rect(fill = "white")) +
  labs(title = 'Residual Histogram', x = 'Residual', y = 'Frequency')

ggsave("residual_histogram.jpg", plot = last_plot(), path = "output/BART/")


# Variable Importance
var_importance <- investigate_var_importance(model2)
var_names <- names(var_importance$avg_var_props)
var_importance_df <- data.frame(
  variable = var_names,
  inc_prop = var_importance$avg_var_props[var_names] * 100
)

# Plot Variable Importance
ggplot(var_importance_df, aes(x = reorder(variable, inc_prop), y = inc_prop, fill = variable)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal(base_family = "sans", base_size = 14) +
  theme(panel.background = element_rect(fill = "white"), legend.position = "none") +
  labs(title = "Variable Importance", x = "Variables", y = "Importance (%)")

ggsave("variable_importance.jpg", plot = last_plot(), path = "output/BART/")


# Save Variable Importance
data.table::fwrite(var_importance_df, paste0(output_path, "BART/BART_var_importance.csv"))

# Disaggregation from Admin Level 2 to 3 ----------------------------------

# Load Admin Level 3 covariates
admin3_covs <- read.csv(paste0(input_path, "covariates/postos/all_features_postos.csv"))

# Load Admin 2-to-3 mapping
admin_mapping <- read.csv(paste0(input_path, "mappings/mozam_admin_2_to_3_mappings.csv"))

# Ensure mappings align with Admin 3 data
admin3_covs <- admin3_covs %>%
  left_join(admin_mapping, by = c("ADM3_PT", "ADM3_PCODE"))


# Remove metadata
covs_admin3 <- admin3_covs %>% select(-ADM3_PT, -ADM3_PCODE, -T_TL, -district_area, -log_population,
                                     -ADM3_PT, -ADM3_PCODE, -ADM3_REF, -ADM3ALT1_PT, -ADM3ALT2_PT, -ADM2_PT, -ADM2_PCODE, -ADM1_PT, -ADM1_PCODE, -ADM0_EN, -ADM0_PT, -ADM0_PCODE, -DATE, -VALIDON, -VALIDTO, -AREA_SQKM)

# Select Covariates - Trees, Built.Area, building_area, building_count, osm_roads, osm_potw
covs_admin3 <- covs_admin3 %>%
  select(Trees, Built.Area, building_area, building_count, osm_roads, osm_pofw, DNB, Water, SR_B6, SR_B1)

#Standardize covs
for (var in names(covs_admin3)) {
  if (var %in% cov_stats$Covariate) {
    var_mean <- cov_stats$Mean[cov_stats$Covariate == var]
    var_sd <- cov_stats$Std_Dev[cov_stats$Covariate == var]
    covs_admin3[[var]] <- (covs_admin3[[var]] - var_mean) / var_sd
  }
}

# Predict population density for Admin Level 3
admin3_predictions <- predict(model2, new_data = covs_admin3)

# Redistribute population to Admin Level 3
admin3_data <- admin3_covs %>%
  mutate(predicted_density = exp(admin3_predictions)) %>%
  group_by(ADM2_PT) %>%
  mutate(
    weight = predicted_density / sum(predicted_density, na.rm = TRUE),
    predicted_population = weight * admin2_data$T_TL[match(ADM2_PT, admin2_data$ADM2_PT)]
  ) %>%
  ungroup()

# Validate totals
admin3_validation <- admin3_data %>%
  group_by(ADM2_PT) %>%
  summarise(total_population = sum(predicted_population, na.rm = TRUE))

all.equal(
  admin3_validation$total_population,
  admin2_data$T_TL[match(admin3_validation$ADM2_PT, admin2_data$ADM2_PT)]
)

# Save results
write.csv(admin3_data, paste0(output_path, "BART/predicted_population_Admin3.csv"), row.names = FALSE)

# Plot Predicted Population Histogram
admin3_data %>%
  ggplot(aes(predicted_population)) +
  geom_histogram(fill = "blue", bins = 30, alpha = 0.7) +
  theme_minimal(base_family = "sans", base_size = 14) +
  theme(panel.background = element_rect(fill = "white")) +
  labs(title = "Predicted Population (Admin Level 3)", x = "Population", y = "Frequency")

ggsave("predicted_population_histogram.jpg", plot = last_plot(), path = "output/BART/")


# Final accuracy checks, predicted admin level 3 population counts vs actual
admin3_data_clean <- admin3_data %>% drop_na(T_TL, predicted_population)
actual_pop <- admin3_data_clean$T_TL
predicted_pop <- admin3_data_clean$predicted_population

# Calculate RMSE
rmse <- sqrt(mean((actual_pop - predicted_pop)^2))

# Calculate MAPE
mape <- mean(abs((actual_pop - predicted_pop) / actual_pop)) * 100

# Calculate R-squared
r_squared <- cor(actual_pop, predicted_pop)^2

# Print results
print("Accuracy Metrics:\n")
print(sprintf("RMSE: %.2f\n", rmse))
print(sprintf("MAPE: %.2f%%\n", mape))
print(sprintf("R-squared: %.3f\n", r_squared))