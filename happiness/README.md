# Dataset Analysis Report

## Insights from LLM

# README.md

## Dataset Analysis

### Overview
This dataset consists of various metrics that provide insights into the well-being and happiness levels across different countries over a span of years. The key parameters include life satisfaction (referred to as "Life Ladder"), economic indicators (like "Log GDP per capita"), and social measures (such as "Social support"). Understanding these factors can help to identify correlations that contribute to overall happiness.

### Table of Contents
1. [Dataset Description](#dataset-description)
2. [Summary Statistics](#summary-statistics)
3. [Missing Values](#missing-values)
4. [Outlier Analysis](#outlier-analysis)
5. [Visualizations](#visualizations)
6. [Key Findings](#key-findings)
7. [References](#references)

### Dataset Description
The dataset includes the following columns:
- **Country name**: The name of the country.
- **year**: The year of the recorded data.
- **Life Ladder**: A metric indicating the general life satisfaction.
- **Log GDP per capita**: The logarithm of GDP per capita, an economic measure.
- **Social support**: A measure of social connections and support systems.
- **Healthy life expectancy at birth**: The average lifespan adjusted for expected quality of life.
- **Freedom to make life choices**: A subjective measure of individual freedom.
- **Generosity**: A measure of how philanthropic a society is perceived to be.
- **Perceptions of corruption**: A subjective assessment of corruption levels in government and business.
- **Positive affect**: The feeling of positive emotions experienced.
- **Negative affect**: The feeling of negative emotions experienced.
- **Cluster**: The result of clustering analysis categorizing countries.

### Summary Statistics
Here are key summary statistics from the dataset:

| Statistic | year | Life Ladder | Log GDP per capita | Positive affect | Negative affect | Cluster |
|-----------|------|-------------|---------------------|-----------------|-----------------|--------|
| Count     | 2363 | 2363        | 2335                | 2339            | 2347            | 2097   |
| Mean      | 2014.76 | 5.48      | 9.40                | 0.65            | 0.27            | 0.49   |
| Std       | 5.06  | 1.13       | 1.15                | 0.11            | 0.09            | 0.50   |
| Min       | 2005  | 1.28       | 5.53                | 0.18            | 0.08            | 0.00   |
| 25%       | 2011  | 4.65       | 8.51                | 0.57            | 0.21            | 0.00   |
| 50%       | 2015  | 5.45       | 9.50                | 0.66            | 0.26            | 0.00   |
| 75%       | 2019  | 6.32       | 10.39               | 0.74            | 0.33            | 1.00   |
| Max       | 2023  | 8.02       | 11.68               | 0.88            | 0.71            | 1.00   |

### Missing Values
The dataset contains missing values in the following columns:
- **Log GDP per capita**: 28 missing values
- **Social support**: 13 missing values
- **Healthy life expectancy at birth**: 63 missing values
- **Freedom to make life choices**: 36 missing values
- **Generosity**: 81 missing values
- **Perceptions of corruption**: 125 missing values
- **Positive affect**: 24 missing values
- **Negative affect**: 16 missing values

### Outlier Analysis
Outliers were identified in the following columns:
- **Log GDP per capita**: 1 outlier
- **Social support**: 32 outliers
- **Healthy life expectancy at birth**: 15 outliers
- **Freedom to make life choices**: 13 outliers
- **Generosity**: 39 outliers
- **Perceptions of corruption**: 184 outliers
- **Positive affect**: 6 outliers
- **Negative affect**: 28 outliers

### Visualizations
The following visualizations were created to explore different aspects of the dataset:

- **Correlation Heatmap**: Displays the correlations between variables. ![Correlation Heatmap](correlation_heatmap.png)
- **Pairplot**: Provides a scatterplot of all numerical variables to identify potential relationships. ![Pairplot](pairplot.png)
- **Regression Plot**: Shows the relationship between two specific variables. ![Regression Plot](regression_plot.png)
- **Histograms**: Displays the distributions of each numeric column (available in the directory).
- **KMeans Clustering**: Clustered data insights are saved as [clustered_data.csv](./clustered_data.csv).

### Key Findings
1. **Positive Correlations**: The analysis indicates strong positive correlations between "Log GDP per capita" and "Life Ladder", suggesting that countries with higher income levels tend to report better life satisfaction.
2. **Negative Affect**: A notable negative correlation exists between "Positive affect" and "Negative affect".
3. **Clustering Insights**: The KMeans clustering reveals distinct groups of countries based on life satisfaction and economic factors, indicating potential socio-economic patterns.

### References
- [Correlation Heatmap](./correlation_heatmap.png)
- [Pairplot](./pairplot.png)
- [Regression Plot](./regression_plot.png)
- [Clustered Data](./clustered_data.csv)

---

This README provides a comprehensive overview of the dataset and analysis undertaken. Insights from the visualizations can aid in understanding how various factors contribute to well-being across countries. Further studies can expand on this foundation for policy-making and social change initiatives.


## Visualizations
- [Correlation Heatmap](./happiness_correlation_heatmap.png)
- [Pairplot](./happiness_pairplot.png)
- [Regression Plot](./happiness_regression_plot.png)
- [Histogram for happiness_year_histogram.png](./happiness_year_histogram.png)
- [Histogram for happiness_Life Ladder_histogram.png](./happiness_Life Ladder_histogram.png)
- [Histogram for happiness_Log GDP per capita_histogram.png](./happiness_Log GDP per capita_histogram.png)
- [Histogram for happiness_Social support_histogram.png](./happiness_Social support_histogram.png)
- [Histogram for happiness_Healthy life expectancy at birth_histogram.png](./happiness_Healthy life expectancy at birth_histogram.png)
- [Histogram for happiness_Freedom to make life choices_histogram.png](./happiness_Freedom to make life choices_histogram.png)
- [Histogram for happiness_Generosity_histogram.png](./happiness_Generosity_histogram.png)
- [Histogram for happiness_Perceptions of corruption_histogram.png](./happiness_Perceptions of corruption_histogram.png)
- [Histogram for happiness_Positive affect_histogram.png](./happiness_Positive affect_histogram.png)
- [Histogram for happiness_Negative affect_histogram.png](./happiness_Negative affect_histogram.png)
- [Clustered Data CSV](./happiness_clustered.csv)
