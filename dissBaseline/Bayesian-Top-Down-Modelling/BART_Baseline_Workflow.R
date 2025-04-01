# Load necessary libraries
library(tidyverse)
options(java.parameters = "-Xmx8g")
library(bartMachine)
library(tictoc)
library(feather)
library(groupdata2)
library(terra)
library(caret)
library(dplyr)
library(ggplot2)


# Specify Paths
input_path <- "data/"  # Adjust path as needed
output_path <- "output/"

# Load Combined Data
admin2_data <- read.csv(paste0(input_path, "covariates/district/all_features_districts.csv"))

# # Function to apply Winsorization (clamping extreme values)
# winsorize <- function(x, lower_percentile = 0.2, upper_percentile = 0.8) {
#   lower_bound <- quantile(x, probs = lower_percentile, na.rm = TRUE)
#   upper_bound <- quantile(x, probs = upper_percentile, na.rm = TRUE)
#   x <- ifelse(x < lower_bound, lower_bound, x)  # Clamp lower values
#   x <- ifelse(x > upper_bound, upper_bound, x)  # Clamp upper values
#   return(x)
# }

# Plot distribution of data
p1 <- ggplot(admin2_data, aes(x = (T_TL / district_area))) +
  geom_histogram(fill = "blue", bins = 40, alpha = 0.7) +
  theme_minimal() +
  labs(
    title = "Original Population Density Distribution", 
    x = "Population Density (People per unit area)", 
    y = "Frequency"
  ) +
  theme(
    plot.title = element_text(size = 18, face = "bold"),
    axis.title = element_text(size = 18, face = "bold"),
    axis.text = element_text(size = 16)
  )

ggsave("PopulationDensityDistributionPlot.jpg", plot = p1, path = "output/BART/", dpi = 500)

p2 <- ggplot(admin2_data, aes(x = log(T_TL / district_area))) +
  geom_histogram(fill = "blue", bins = 40, alpha = 0.7) +
  theme_minimal() +
  labs(
    title = "Log Population Density Distribution", 
    x = "Log Population Density (People per unit area)", 
    y = "Frequency"
  ) +
  theme(
    plot.title = element_text(size = 18, face = "bold"),
    axis.title = element_text(size = 18, face = "bold"),
    axis.text = element_text(size = 16)
  )

ggsave("LogPopulationDensityDistributionPlot.jpg", plot = p2, path = "output/BART/", dpi = 500)


#Remove population counts of 0
admin2_data <- admin2_data %>%
  filter(T_TL > 0)

# Define response variable as log of population density
admin2_data <- admin2_data %>%
  mutate(pop_density = log(T_TL / district_area)) %>%
  filter(!is.na(pop_density))

# admin2_data <- admin2_data %>%
#   mutate(pop_density = winsorize(pop_density))

# Plot distribution of data
ggplot(admin2_data, aes(x = (pop_density))) +
  geom_histogram(fill = "blue", bins = 30, alpha = 0.7) +
  theme_minimal() +
  labs(title = "Original Population Density Distribution Clamped", 
       x = "Population Density (People per unit area)", 
       y = "Frequency")
ggsave("PopulationDensityDistributionPlotClamped.jpg", plot = last_plot(), path = "output/BART/")


#Ensure OSM covariates are in density units
admin2_data <- admin2_data %>%
  mutate(across(starts_with("osm"), ~ . / district_area))

#Ensure building_count covariates are in density units
admin2_data <- admin2_data %>%
  mutate(building_count = building_count / district_area)

admin2_data <- admin2_data %>%
  mutate(building_area = building_area / district_area)

# Select covariates (exclude identifiers and response-related columns)
covs_admin2 <- admin2_data %>%
  select(-ADM2_PT, -ADM2_PCODE, -T_TL, -district_area, -pop_density, -log_population)


# Select Covariates - Trees, Built.Area, building_area, building_count, osm_roads, osm_potw
# covs_admin2 <- covs_admin2 %>%
#   select(Trees, Built.Area, building_area, building_count, osm_roads, osm_pofw, DNB, Water, SR_B6, SR_B1)

covs_admin2 <- covs_admin2 %>%
  select(Trees, Built.Area, building_area, building_count, osm_roads, osm_pois, osm_traffic, osm_transport, osm_railways, osm_pofw, DNB)

# Calculate mean and standard deviation of covariates for scaling
cov_stats <- data.frame(
  Covariate = colnames(covs_admin2),
  Mean = apply(covs_admin2, 2, mean, na.rm = TRUE),
  Std_Dev = apply(covs_admin2, 2, sd, na.rm = TRUE)
)

admin2_means <- cov_stats$Mean
names(admin2_means) <- cov_stats$Covariate

admin2_sds <- cov_stats$Std_Dev
names(admin2_sds) <- cov_stats$Covariate

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

set.seed(123)
#use 70% of dataset as training set and 30% as test set
smp_size <- floor(0.8 * nrow(admin2_data_scaled))

## set the seed to make your partition reproducible
set.seed(123)
sample <- sample(seq_len(nrow(admin2_data_scaled)), size = smp_size)

train  <- admin2_data_scaled[sample, ]
train_covs <- train %>% select(all_of(colnames(covs_admin2)))
test   <- admin2_data_scaled[-sample, ]
test_covs <- test %>% select(all_of(colnames(covs_admin2)))

################ Hyperparameter search
# Define the hyperparameter grid
# hyper_grid <- expand.grid(
#   num_trees = c(100, 200, 500, 700),  # Number of trees
#   k = c(2, 3, 4),  # Regularization parameter
#   nu = c(3, 5, 10),  # Shape of variance prior
#   q = c(0.25, 0.5, 0.75)  # Quantile of the variance prior
# )



# # Initialize results dataframe
# results <- data.frame()

# # Loop over each hyperparameter combination
# for (i in 1:nrow(hyper_grid)) {
#   set.seed(1234)  # Ensure reproducibility
  
#   # Train the BART model with current parameters
#   model <- bartMachine(
#     X = train_covs, 
#     y = train$pop_density,
#     num_trees = hyper_grid$num_trees[i],
#     k = hyper_grid$k[i],
#     nu = hyper_grid$nu[i],
#     q = hyper_grid$q[i],
#     use_missing_data = TRUE,
#     verbose = FALSE  # Suppress output for cleaner execution
#   )
  
#   # Extract Out-of-Bag RMSE as the evaluation metric
#   oob_rmse <- model$rmse
#   print(oob_rmse)
  
#   # Store results
#   new_row <- data.frame(hyper_grid[i, ], oob_rmse = oob_rmse)
#   results <- rbind(results, new_row)
# }

# Find the best hyperparameter combination
# best_params <- results[which.min(results$oob_rmse), ]

# # Print the best hyperparameters
# print("Best Hyperparameters for BART:")
# print(best_params)

################## Hyperparameter search

# Train the BART Model
set.seed(1234)
model2 <- bartMachine(
  X = train_covs,
  y = train$pop_density,
  k = 1, nu = 3, q = 0.25, num_trees = 500, use_missing_data = TRUE
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
    observed = exp(train$pop_density),
    predicted = exp(value),
    residual = observed - predicted,
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
    mape = mean(abs((observed - predicted) / observed)) * 100,
    corr = cor(predicted, observed),
    In_IC = mean(observed < ci_lower_bd & observed > ci_upper_bd) * 100,
    r2 = 1 - (sum((observed - predicted)^2) / 
    sum((observed - mean(observed))^2))
  )

# Print the metrics
print("Goodness-of-fit metrics Train:")
print(admin2_metrics)

# Save metrics to a CSV file
write.csv(admin2_metrics, paste0(output_path, "BART/model_metrics.csv"), row.names = FALSE)

# Print the admin2_predictions for verification
# print("Admin Level 2 Predictions:")
# print(admin2_predictions)

#Predictions on test data
predicted_values <- predict(model2, new_data = test_covs)

# Create a data frame for predictions and observed values
admin2_predictions_test <- data.frame(
  predicted = exp(predicted_values),
  observed = exp(test$pop_density)
)

# Compute credible intervals
admin2_CI_test <- calc_credible_intervals(model2, new_data = test_covs) %>%
  as_tibble() %>%
  mutate(ci_lower_bd = ci_lower_bd, ci_upper_bd = ci_upper_bd)

# Combine predictions with observed values
admin2_predictions_test <- admin2_predictions_test %>%
  cbind(test$pop_density, admin2_CI_test) %>%
  mutate(
    residual = observed - predicted,
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
    mape = mean(abs((observed - predicted) / observed)) * 100,
    corr = cor(predicted, observed),
    In_IC = mean(observed < ci_lower_bd & observed > ci_upper_bd) * 100,
    r2 = 1 - (sum((observed - predicted)^2) / 
    sum((observed - mean(observed))^2))
  )

# Print the metrics
print("Goodness-of-fit metrics Test:")
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
p <- ggplot(var_importance_df, aes(x = reorder(variable, inc_prop), y = inc_prop, fill = variable)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(size = 20, face = "bold"),
    axis.title = element_text(size = 20, face = "bold"),
    axis.text = element_text(size = 20),
    panel.background = element_rect(fill = "white"),
    legend.position = "none"
  ) +
  labs(
    title = "Variable Importance", 
    x = "Variables", 
    y = "Importance (%)"
  )

# Save with high resolution
ggsave("variable_importance.jpg", plot = p, path = "output/BART/", dpi = 500, width = 12, height = 12)


# Save Variable Importance
data.table::fwrite(var_importance_df, paste0(output_path, "BART/BART_var_importance.csv"))

# Disaggregation from Admin Level 2 to 3 ----------------------------------

# Load Admin Level 3 covariates
admin3_covs <- read.csv(paste0(input_path, "covariates/postos/all_features_postos.csv"))

#Remove population counts of 0
admin3_covs <- admin3_covs %>%
  filter(T_TL > 0)

# Load Admin 2-to-3 mapping
admin_mapping <- read.csv(paste0(input_path, "mappings/mozam_admin_2_to_3_mappings.csv"))

# Ensure mappings align with Admin 3 data
admin3_covs <- admin3_covs %>%
  left_join(admin_mapping, by = c("ADM3_PT", "ADM3_PCODE"))


#Ensure OSM covariates are in density units
admin3_covs <- admin3_covs %>%
  mutate(across(starts_with("osm"), ~ . / district_area))

#Ensure building_count covariates are in density units
admin3_covs <- admin3_covs %>%
  mutate(building_count = building_count / district_area)

admin3_covs <- admin3_covs %>%
  mutate(building_area = building_area / district_area)

# Remove metadata
# covs_admin3 <- admin3_covs %>% select(-ADM3_PT, -ADM3_PCODE, -T_TL, -district_area, -log_population,
#                                      -ADM3_PT, -ADM3_PCODE, -ADM3_REF, -ADM3ALT1_PT, -ADM3ALT2_PT, -ADM2_PT, -ADM2_PCODE, -ADM1_PT, -ADM1_PCODE, -ADM0_EN, -ADM0_PT, -ADM0_PCODE, -DATE, -VALIDON, -VALIDTO, -AREA_SQKM)

# Select Covariates - Trees, Built.Area, building_area, building_count, osm_roads, osm_potw
# covs_admin3 <- covs_admin3 %>%
#   select(Trees, Built.Area, building_area, building_count, osm_roads, osm_pois, osm_traffic, osm_transport, osm_railways, osm_pofw, DNB)

covs_admin3 <- admin3_covs %>%
  select(Trees, Built.Area, building_area, building_count, osm_roads, osm_pois, 
         osm_traffic, osm_transport, osm_railways, osm_pofw, DNB) %>%
  mutate(across(
    everything(),
    ~ (. - admin2_means[cur_column()]) / admin2_sds[cur_column()]
  ))

# Calculate mean and standard deviation of covariates for scaling
# cov3_stats <- data.frame(
#   Covariate = colnames(covs_admin3),
#   Mean = apply(covs_admin3, 2, mean, na.rm = TRUE),
#   Std_Dev = apply(covs_admin3, 2, sd, na.rm = TRUE)
# )

# Apply scaling function
# covs_admin3 <- apply(covs_admin3, 2, stdize) %>% as_tibble()

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
actual_density <- admin3_data_clean$T_TL / admin3_data_clean$district_area
predicted_pop_disag <- admin3_data_clean$predicted_population
predicted_density <- admin3_data_clean$predicted_density

# Calculate MAE
mae_dyas <- mean(abs(actual_pop - predicted_pop_disag))

# Calculate MAPE
mape_dyas <- mean(abs((actual_pop - predicted_pop_disag) / actual_pop)) * 100

# Calculate R-squared
r_squared_dyas <- cor(actual_pop, predicted_pop_disag)^2

# Print results
print("Accuracy Metrics Dyasymetric (WorldPop Admin 3):")

print(sprintf("MAE: %.2f", mae_dyas))
print(sprintf("MAPE: %.2f%%", mape_dyas))
print(sprintf("R-squared: %.3f", r_squared_dyas))
cat("\n")

# Calculate MAE
mae <- mean(abs(actual_density - predicted_density))

# Calculate MAPE
mape <- mean(abs((actual_density - predicted_density) / actual_density)) * 100

# Calculate R-squared
r_squared <- cor(actual_density, predicted_density)^2

# Print results
print("Accuracy Metrics Normal (WorldPop Admin 3):")
print(sprintf("MAE: %.2f", mae))
print(sprintf("MAPE: %.2f%%", mape))
print(sprintf("R-squared: %.3f", r_squared))