# Load necessary libraries
library(tidyverse)
library(randomForest)
library(tictoc)
library(feather)
library(groupdata2)
library(terra)
library(caret)
library(lsa)
library(e1071)

# Specify Paths
input_path <- "data/"  # Adjust path as needed
output_path <- "output/"

# Load Combined Data
admin2_data <- read.csv(paste0(input_path, "covariates/district/all_features_districts.csv"))

#Remove population counts of 0
admin2_data <- admin2_data %>%
  filter(T_TL > 0)

# Define response variable as log of population density
admin2_data <- admin2_data %>%
  mutate(pop_density = log(T_TL / district_area)) %>%
  filter(!is.na(pop_density))


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
#   select(Trees, Built.Area, building_area, building_count, osm_roads, osm_pois, osm_traffic, osm_transport, osm_pofw)

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

print(paste0("Skew: ", skewness(admin2_data_scaled$pop_density)))

# mtry <- tuneRF(x = train_covs, y = train$pop_density, na.action = na.omit,
#        plot = T, trace = T, importance=TRUE,  sampsize=length(train), replace=TRUE)[,1]

# optimalMtry <- mtry[which.min(mtry[,2]), 1]




################################# Hyperparameter searching
# hyper_grid <- expand.grid(
#   mtry = optimal_mtry,
#   ntree = c(100, 300, 500, 1000),   
#   nodesize = c(1, 5, 10),           
#   sampsize = c(0.5, 0.7, 1.0) * nrow(train)
# )

# # Init
# results <- data.frame()


# for (i in 1:nrow(hyper_grid)) {
#   set.seed(123)
  
#   model <- randomForest(
#     x = train_covs, 
#     y = train$pop_density, 
#     mtry = hyper_grid$mtry[i], 
#     ntree = hyper_grid$ntree[i], 
#     nodesize = hyper_grid$nodesize[i],
#     sampsize = hyper_grid$sampsize[i],
#     na.action = na.omit,
#     importance = TRUE
#   )
  
#   # Store results
#   oob_error <- model$mse[length(model$mse)]  # Final OOB error
#   results <- rbind(results, cbind(hyper_grid[i,], oob_error))
# }

# # Find best hyperparameter combination
# best_params <- results[which.min(results$oob_error), ]
# print("Best: ", str(best_params))
##################### Hyperparameter search




# #Fit model
# model2 <- randomForest(x = train_covs, y = train$pop_density, mtry = min(ncol(train_covs), 40), na.action = na.omit, 
#                       plot = T, trace = T, importance=TRUE, sampsize=length(train), replace=TRUE) 

#Fit Best Model
model2 <- randomForest(x = train_covs, y = train$pop_density, mtry = 8, na.action = na.omit, ntree = 300, nodesize = 1, 
                      plot = T, trace = T, importance=TRUE, sampsize=108, replace=TRUE) 


# Predictions on training data
predicted_values <- predict(model2, newdata = train_covs)

# Create a tibble combining predictions and observed values
admin2_predictions <- tibble(
  observed = exp(train$pop_density),
  predicted = exp(predicted_values),
  residual = observed - predicted,
  model = "RF"
)

# Save results
write.csv(admin2_predictions, paste0(output_path, "RF/RF_model_results.csv"))

# Goodness-of-fit metrics
admin2_metrics <- admin2_predictions %>%
  summarise(
    Bias = mean(residual),
    Imprecision = sd(residual),
    Inaccuracy = mean(abs(residual)),
    mse = mean(residual^2),
    mape = mean(abs((observed - predicted) / observed)) * 100,
    rmse = sqrt(mse),
    corr = cor(predicted, observed),
    r2 = 1 - (sum((observed - predicted)^2) / 
    sum((observed - mean(observed))^2))
  )

# Print the metrics
print("Goodness-of-fit metrics Train:")
print(admin2_metrics)

# Save metrics to a CSV file
write.csv(admin2_metrics, paste0(output_path, "RF/model_metrics_train.csv"), row.names = FALSE)

# Print the admin2_predictions for verification
# print("Admin Level 2 Predictions:")
# print(admin2_predictions)

# Convergence plot
# oob_error <- model2$err.rate[, "OOB"]

# # 6. Plot OOB error vs. number of trees
# plot(oob_error, type = "l", 
#      xlab = "Number of Trees",
#      ylab = "OOB Error Rate",
#      main = "OOB Error vs. Number of Trees (Random Forest)")

# Alternatively, randomForest has a built-in plot method:
# This will show OOB error vs. trees (and class-specific error if multi-class).
png("output/RF/oob_error_convergence_plot.png", width=600, height=400)
plot(model2$mse, type="l", xlab="Number of Trees", ylab="OOB MSE")
dev.off()
#Predictions on test data
predicted_values <- predict(model2, newdata = test_covs)

# Create a data frame for predictions and observed values
admin2_predictions_test <- data.frame(
  predicted = exp(predicted_values),
  observed = exp(test$pop_density)
)

# Combine predictions with observed values
admin2_predictions_test <- admin2_predictions_test %>%
  mutate(
    residual = observed - predicted,
    model = "RF"
  )

# Save results
write.csv(admin2_predictions_test, paste0(output_path, "RF/RF_model_results.csv"))

# Goodness-of-fit metrics
admin2_metrics_test <- admin2_predictions_test %>%
  summarise(
    Bias = mean(residual),
    Imprecision = sd(residual),
    Inaccuracy = mean(abs(residual)),
    mse = mean(residual^2),
    mape = mean(abs((observed - predicted) / observed)) * 100,
    rmse = sqrt(mse),
    corr = cor(predicted, observed),
    r2 = 1 - (sum((observed - predicted)^2) / 
    sum((observed - mean(observed))^2))
  )

# Print the metrics
print("Goodness-of-fit metrics Test:")
print(admin2_metrics_test)

# Save metrics to a CSV file
write.csv(admin2_metrics_test, paste0(output_path, "RF/model_metrics_test.csv"), row.names = FALSE)


# Plot Observed vs. Predicted
library(ggplot2)
ggplot(admin2_predictions,
    aes(x = observed, y = predicted),
    fill = 'grey50', color = 'grey70', shape = 21
  ) +
  geom_abline(slope = 1, intercept = 0, color = 'orange', linewidth = 1) +
  theme_minimal(base_family = "sans", base_size = 14) +
  theme(panel.background = element_rect(fill = "white")) +
  labs(title = '', x = 'Observed Population Density', y = 'Predicted Density')

ggsave("observed_vs_predicted.jpg", plot = last_plot(), path = "output/RF/")

# Plot Residual Histogram
admin2_predictions %>%
  ggplot(aes(residual)) +
  geom_histogram(fill = 'blue', bins = 30, alpha = 0.7) +
  theme_minimal(base_family = "sans", base_size = 14) +
  theme(panel.background = element_rect(fill = "white")) +
  labs(title = 'Residual Histogram', x = 'Residual', y = 'Frequency')

ggsave("residual_histogram.jpg", plot = last_plot(), path = "output/RF/")


# Variable Importance
varImpPlot(model2 , sort = TRUE , n.var = 10 , main = "Variable Importance Random Forest")

ggsave("variable_importance.jpg", plot = last_plot(), path = "output/RF/")


# Save Variable Importance
data.table::fwrite(var_importance_df, paste0(output_path, "RF/RF_var_importance.csv"))

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
# covs_admin3 <- covs_admin3 %>%
#   select(Trees, Built.Area, building_area, building_count, osm_roads, osm_pofw, DNB, Water, SR_B6, SR_B1)

#Standardize covs
for (var in names(covs_admin3)) {
  if (var %in% cov_stats$Covariate) {
    var_mean <- cov_stats$Mean[cov_stats$Covariate == var]
    var_sd <- cov_stats$Std_Dev[cov_stats$Covariate == var]
    covs_admin3[[var]] <- (covs_admin3[[var]] - var_mean) / var_sd
  }
}

# Predict population density for Admin Level 3
admin3_predictions <- predict(model2, newdata = covs_admin3)

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
write.csv(admin3_data, paste0(output_path, "RF/predicted_population_Admin3.csv"), row.names = FALSE)

# Plot Predicted Population Histogram
admin3_data %>%
  ggplot(aes(predicted_population)) +
  geom_histogram(fill = "blue", bins = 30, alpha = 0.7) +
  theme_minimal(base_family = "sans", base_size = 14) +
  theme(panel.background = element_rect(fill = "white")) +
  labs(title = "Predicted Population (Admin Level 3)", x = "Population", y = "Frequency")

ggsave("predicted_population_histogram.jpg", plot = last_plot(), path = "output/RF/")


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