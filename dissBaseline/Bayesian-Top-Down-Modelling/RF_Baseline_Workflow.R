# Load necessary libraries
library(tidyverse)
library(randomForest)
library(tictoc)
library(feather)
library(groupdata2)
library(terra)

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

# Tune Random Forest Model
tuneRF(
  x = covs_admin2, y = admin2_data_scaled$pop_density, 
  na.action = na.omit, plot = TRUE, trace = TRUE, importance = TRUE, 
  sampsize = nrow(admin2_data_scaled), replace = TRUE
)

# Train Random Forest Model
set.seed(1234)
model_rf <- randomForest(
  x = covs_admin2, y = admin2_data_scaled$pop_density, 
  mtry = 16, na.action = na.omit, importance = TRUE, 
  sampsize = nrow(admin2_data_scaled), replace = TRUE
)

# Extract Predictions
admin2_rf_predictions <- data.frame(
  observed = admin2_data_scaled$pop_density,
  predicted = predict(model_rf),
  residual = predict(model_rf) - admin2_data_scaled$pop_density
)

# Save Random Forest Predictions
write.csv(admin2_rf_predictions, paste0(output_path, "RF/RF_model_results.csv"), row.names = FALSE)

# Calculate Goodness-of-Fit Metrics
admin2_rf_metrics <- admin2_rf_predictions %>%
  summarise(
    Bias = mean(residual),
    Imprecision = sd(residual),
    Inaccuracy = mean(abs(residual)),
    mse = mean(residual^2),
    rmse = sqrt(mse),
    corr = cor(predicted, observed)
  )

# Print Metrics and Save to File
print("Random Forest Goodness-of-Fit Metrics:")
print(admin2_rf_metrics)
write.csv(admin2_rf_metrics, paste0(output_path, "RF/RF_metrics.csv"), row.names = FALSE)

# Plot Observed vs. Predicted
observed_vs_predicted_plot <- ggplot(admin2_rf_predictions) +
  geom_point(aes(x = observed, y = predicted), color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'orange', linewidth = 1) +
  theme_minimal(base_family = "sans", base_size = 14) +
  theme(panel.background = element_rect(fill = "white")) +
  labs(title = "Random Forest: Observed vs. Predicted", 
       x = "Observed Population Density", y = "Predicted Population Density")

ggsave(filename = paste0(output_path, "RF/observed_vs_predicted_rf.jpg"), plot = observed_vs_predicted_plot, width = 8, height = 6)

# Plot Residual Histogram
residual_histogram_plot <- admin2_rf_predictions %>%
  ggplot(aes(residual)) +
  geom_histogram(fill = 'blue', bins = 30, alpha = 0.7) +
  theme_minimal(base_family = "sans", base_size = 14) +
  theme(panel.background = element_rect(fill = "white")) +
  labs(title = "Residual Histogram (Random Forest)", x = "Residual", y = "Frequency")

ggsave(filename = paste0(output_path, "RF/residual_histogram_rf.jpg"), plot = residual_histogram_plot, width = 8, height = 6)

# Plot Variable Importance
rf_var_importance <- data.frame(
  Variable = rownames(importance(model_rf)),
  Importance = importance(model_rf)[, "IncNodePurity"]
)

variable_importance_plot <- ggplot(rf_var_importance, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  theme_minimal(base_family = "sans", base_size = 14) +
  theme(panel.background = element_rect(fill = "white")) +
  labs(title = "Variable Importance (Random Forest)", x = "Variable", y = "Importance")

ggsave(filename = paste0(output_path, "RF/variable_importance_rf.jpg"), plot = variable_importance_plot, width = 8, height = 6)

# Save Variable Importance to File
write.csv(rf_var_importance, paste0(output_path, "RF/RF_var_importance.csv"), row.names = FALSE)

# Plot Predicted Population Histogram
predicted_population_histogram <- admin2_rf_predictions %>%
  ggplot(aes(predicted)) +
  geom_histogram(fill = "blue", bins = 30, alpha = 0.7) +
  theme_minimal(base_family = "sans", base_size = 14) +
  theme(panel.background = element_rect(fill = "white")) +
  labs(title = "Predicted Population Density Histogram (Random Forest)", 
       x = "Predicted Population Density", y = "Frequency")

ggsave(filename = paste0(output_path, "RF/predicted_population_histogram_rf.jpg"), 
       plot = predicted_population_histogram, width = 8, height = 6)
