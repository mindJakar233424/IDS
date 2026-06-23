# ============================================
# PHASE A: DATA UNDERSTANDING
# ============================================

# --------------------------
# 1. Install Required Packages
# --------------------------
install.packages(c(
  "readr", "dplyr", "tibble", "psych", "skimr","moments",
  "ggplot2", "corrplot", "ggcorrplot", "GGally","tidyverse",
  "caret", "fastDummies", "scales", "FSelector","gridExtra"
))

# --------------------------
# 2. Load Libraries
# --------------------------
library(tidyverse)
library(dplyr)
library(readr)
library(skimr)
library(psych)
library(moments)
library(ggplot2)
library(gridExtra)
library(ggcorrplot)
library(GGally)
library(fastDummies)
library(scales)
library(caret)
library(corrplot)

# --------------------------
# A.1 Load Dataset
# --------------------------
# Load dataset using base R
blood_data <- read.csv("https://raw.githubusercontent.com/mindJakar233424/IDS/refs/heads/main/blood_data.csv")



cat("\n=== First 10 Rows of Dataset ===\n")
head(blood_data, 10)

# --------------------------
# A.2 Dataset Structure & Shape
# --------------------------
cat("\n=== Dataset Structure & Shape ===\n")
cat("Rows:", nrow(blood_data), "\n")
cat("Columns:", ncol(blood_data), "\n\n")
str(blood_data)

# --------------------------
# A.3 Data Type Analysis
# --------------------------
cat("\n=== Data Types ===\n")
data_types <- sapply(blood_data, class)
print(data_types)

cat("\n=== Target Variable Distribution (Blood_Group) ===\n")
print(table(blood_data$Blood_Group))
print(prop.table(table(blood_data$Blood_Group)) * 100)

# --------------------------
# A.4 Descriptive Statistics
# --------------------------
cat("\n=== Summary Statistics ===\n")
summary(blood_data)

cat("\n=== Detailed Numerical Statistics ===\n")
numeric_data <- blood_data[, sapply(blood_data, is.numeric)]
describe(numeric_data)


cat("\n=== Categorical Feature Frequencies ===\n")
categorical_cols <- names(blood_data)[sapply(blood_data, function(x) is.character(x) | is.factor(x))]
for(col in categorical_cols) {
  cat(sprintf("\n%s:\n", col))
  print(table(blood_data[[col]]))
}

# --------------------------
# A.5 Feature Type Identification
# --------------------------
cat("\n=== Feature Types ===\n")
numerical_features <- names(blood_data)[sapply(blood_data, is.numeric)]
categorical_features <- names(blood_data)[sapply(blood_data, function(x) is.character(x) | is.factor(x))]

cat("\nNumerical Features:", length(numerical_features), "\n")
print(numerical_features)

cat("\nCategorical Features:", length(categorical_features), "\n")
print(categorical_features)

cat("\nUnique Values per Feature:\n")
print(sapply(blood_data, function(x) length(unique(x))))

# ============================================
# PHASE B: DATA EXPLORATION & VISUALIZATION
# ============================================

# --------------------------
# B.1 Univariate Analysis
# --------------------------

# Identify numeric & categorical columns
numeric_cols <- names(blood_data)[sapply(blood_data, is.numeric)]
categorical_cols <- names(blood_data)[sapply(blood_data, function(x) is.character(x) | is.factor(x))]

# Histograms for Numeric Features
cat("\n=== Creating Histograms for Numeric Features ===\n")
hist_list <- list()
for(col in numeric_cols){
  p <- ggplot(blood_data, aes(x = !!sym(col))) +
    geom_histogram(bins = 25, fill = "steelblue", color = "black") +
    ggtitle(paste("Histogram of", col)) +
    theme_minimal()
  hist_list[[col]] <- p
}
grid.arrange(grobs = hist_list[1:min(4, length(hist_list))], ncol = 2)

# Boxplots for Numeric Features
cat("\n=== Boxplots for Numeric Features ===\n")
box_list <- list()
for(col in numeric_cols){
  p <- ggplot(blood_data, aes(y = !!sym(col))) +
    geom_boxplot(fill = "orange") +
    ggtitle(paste("Boxplot of", col)) +
    theme_minimal()
  box_list[[col]] <- p
}
grid.arrange(grobs = box_list[1:min(4, length(box_list))], ncol = 2)

# Bar Charts for Categorical Features
cat("\n=== Bar Charts for Categorical Features ===\n")
for(col in categorical_cols){
  print(
    ggplot(blood_data, aes(x = !!sym(col))) +
      geom_bar(fill = "darkred") +
      ggtitle(paste("Frequency of", col)) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  )
}

# Frequency tables
for(col in categorical_cols){
  cat("\nFrequency table for:", col, "\n")
  print(table(blood_data[[col]]))
}

# --------------------------
# B.2 Bivariate Analysis
# --------------------------

# Correlation Matrix for Numeric Columns
cat("\n=== Correlation Matrix ===\n")
if(length(numeric_cols) > 1){
  cor_matrix <- cor(blood_data[numeric_cols], use = "complete.obs")
  print(cor_matrix)
  corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black")
}

# Scatter Plots (first few numeric pairs)
num_pairs <- combn(numeric_cols, 2, simplify = FALSE)
sp_list <- list()
for(i in 1:min(4, length(num_pairs))){
  pair <- num_pairs[[i]]
  p <- ggplot(blood_data, aes(x = !!sym(pair[1]), y = !!sym(pair[2]))) +
    geom_point(alpha = 0.7, color = "black") +
    ggtitle(paste(pair[1], "vs", pair[2])) +
    theme_minimal()
  sp_list[[i]] <- p
}
if(length(sp_list) > 0) grid.arrange(grobs = sp_list, ncol = 2)

# Boxplots Categorical vs Numeric
if(length(categorical_cols) > 0 && length(numeric_cols) > 0){
  for(cat_col in categorical_cols[1:min(2, length(categorical_cols))]){
    for(num_col in numeric_cols[1:min(2, length(numeric_cols))]){
      p <- ggplot(blood_data, aes(x = !!sym(cat_col), y = !!sym(num_col), fill = !!sym(cat_col))) +
        geom_boxplot() +
        ggtitle(paste(num_col, "by", cat_col)) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      print(p)
    }
  }
}

# --------------------------
# B.3 Skewness & Outliers
# --------------------------
cat("\n=== Skewness of Numeric Columns ===\n")
print(sapply(blood_data[numeric_cols], skewness, na.rm = TRUE))

detect_outliers <- function(x){
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  return(which(x < lower | x > upper))
}

outlier_counts <- sapply(blood_data[numeric_cols], detect_outliers)
cat("\nOutlier Counts per Numeric Column:\n")
print(outlier_counts)

# ============================================
# PHASE C: DATA PREPROCESSING
# ============================================

# --------------------------
# C.1 Handling Missing Values
# --------------------------
cat("\n=== Handling Missing Values ===\n")
numeric_vars <- names(blood_data)[sapply(blood_data, is.numeric)]
cat_vars <- names(blood_data)[sapply(blood_data, function(x) is.character(x) | is.factor(x))]

# Replace numeric NAs with median
blood_data[numeric_vars] <- lapply(blood_data[numeric_vars], function(x){
  x[is.na(x)] <- median(x, na.rm = TRUE)
  return(x)
})

# Replace categorical NAs with mode
mode_value <- function(x){
  ux <- unique(x[!is.na(x)])
  ux[which.max(tabulate(match(x, ux)))]
}
for(col in cat_vars){
  blood_data[[col]][is.na(blood_data[[col]])] <- mode_value(blood_data[[col]])
}

cat("\nMissing Values After Imputation:\n")
print(colSums(is.na(blood_data)))

# --------------------------
# C.2 Outlier Treatment
# --------------------------
cap_outliers <- function(x){
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  x[x < lower] <- lower
  x[x > upper] <- upper
  return(x)
}

blood_data[numeric_vars] <- lapply(blood_data[numeric_vars], cap_outliers)

# --------------------------
# C.3 Data Conversion
# --------------------------

# Label Encoding
label_encode <- function(x) as.numeric(factor(x))
blood_data[cat_vars] <- lapply(blood_data[cat_vars], label_encode)

# One-Hot Encoding
valid_cat <- cat_vars[sapply(blood_data[cat_vars], function(x) length(unique(x)) > 1)]
if(length(valid_cat) > 0){
  dummies <- as.data.frame(model.matrix(~ . -1, data = blood_data[valid_cat]))
  blood_data_one_hot <- cbind(blood_data[!names(blood_data) %in% valid_cat], dummies)
} else {
  blood_data_one_hot <- blood_data
}

# --------------------------
# C.4 Data Transformation
# --------------------------
# Min-Max Normalization
min_max_normalize <- function(x) (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
blood_data_minmax <- blood_data
blood_data_minmax[numeric_vars] <- lapply(blood_data_minmax[numeric_vars], min_max_normalize)

# Z-score Standardization
zscore_standardize <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
blood_data_zscore <- blood_data
blood_data_zscore[numeric_vars] <- lapply(blood_data_zscore[numeric_vars], zscore_standardize)

# Log Transformation (only positive values)
blood_data_log <- blood_data
for(col in numeric_vars){
  if(all(blood_data[[col]] > 0, na.rm = TRUE)){
    blood_data_log[[col]] <- log(blood_data[[col]])
  }
}

# Square Root Transformation (non-negative)
blood_data_sqrt <- blood_data
for(col in numeric_vars){
  if(all(blood_data[[col]] >= 0, na.rm = TRUE)){
    blood_data_sqrt[[col]] <- sqrt(blood_data[[col]])
  }
}

# --------------------------
# C.5 Feature Selection
# --------------------------

# Correlation-Based Feature Removal (threshold 0.80)
corr_matrix <- cor(blood_data[numeric_vars], use = "complete.obs")
high_corr <- findCorrelation(corr_matrix, cutoff = 0.80)
cat("Highly Correlated Columns to Remove:\n")
print(colnames(blood_data[numeric_vars])[high_corr])
reduced_corr_data <- blood_data[numeric_vars[-high_corr]]

# Variance Thresholding (>0.1)
variances <- apply(blood_data[numeric_vars], 2, var, na.rm = TRUE)
high_variance_data <- blood_data[numeric_vars[, variances > 0.1]]
cat("\nSelected Features After Variance Thresholding:\n")
print(names(high_variance_data))
