# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "ipykernel",  # Added ipykernel
# ]
# ///

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import openai  # Make sure you install this library: pip install openai

# Function to analyze the data (basic summary stats, missing values, correlation matrix)
def analyze_data(df):
    print("Analyzing the data...")  # Debugging line
    # Summary statistics for numerical columns
    summary_stats = df.describe()

    # Check for missing values
    missing_values = df.isnull().sum()

    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])

    # Correlation matrix for numerical columns
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()

    print("Data analysis complete.")  # Debugging line
    return summary_stats, missing_values, corr_matrix


# Function to detect outliers using the IQR method
def detect_outliers(df):
    print("Detecting outliers...")  # Debugging line
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Apply the IQR method to find outliers in the numeric columns
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()

    print("Outliers detection complete.")  # Debugging line
    return outliers


# Function to generate visualizations (correlation heatmap, outliers plot, and distribution plot)
def visualize_data(corr_matrix, outliers, df, output_dir):
    print("Generating visualizations...")  # Debugging line
    # Generate a heatmap for the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    heatmap_file = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(heatmap_file)
    plt.close()  # Close the figure to avoid displaying it

    # Check if there are outliers to plot
    if not outliers.empty and outliers.sum() > 0:
        # Plot the outliers
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        outliers_file = os.path.join(output_dir, 'outliers.png')
        plt.savefig(outliers_file)
        plt.close()  # Close the figure to avoid displaying it
    else:
        print("No outliers detected to visualize.")
        outliers_file = None  # No file created for outliers

    # Generate a distribution plot for the first numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        first_numeric_column = numeric_columns[0]  # Get the first numeric column
        plt.figure(figsize=(10, 6))
        sns.histplot(df[first_numeric_column], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {first_numeric_column}')
        plt.xlabel(first_numeric_column)
        plt.ylabel('Frequency')
        dist_plot_file = os.path.join(output_dir, 'distribution_plot.png')
        plt.savefig(dist_plot_file)
        plt.close()  # Close the figure to avoid displaying it
    else:
        print("No numeric columns available for distribution plot.")
        dist_plot_file = None  # No file created for distribution plot

    print("Visualizations generated successfully.")  # Debugging line
    return heatmap_file, outliers_file, dist_plot_file


# Function to create the README.md with a narrative and visualizations
def create_readme(summary_stats, missing_values, corr_matrix, out liers, output_dir):
    readme_content = f"""
# Data Analysis Project

This project performs data analysis and visualization on a given dataset. It includes functions for analyzing data, detecting outliers, and generating visualizations.

## Summary Statistics
{summary_stats.to_string()}

## Missing Values
{missing_values.to_string()}

## Correlation Matrix
![Correlation Matrix](correlation_matrix.png)

## Outliers Detection
![Outliers Detection](outliers.png)

## Distribution Plot
![Distribution Plot](distribution_plot.png)

## Requirements

- Python >= 3.9
- pandas
- seaborn
- matplotlib
- numpy
- scipy
- openai
- scikit-learn
- requests
- ipykernel

## Usage

1. Load your dataset into a pandas DataFrame.
2. Call the `analyze_data(df)` function to get summary statistics and correlation matrix.
3. Use `detect_outliers(df)` to find outliers in your data.
4. Generate visualizations using `visualize_data(corr_matrix, outliers, df, output_dir)`.
5. Create a README file with `create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir)`.

## License

This project is licensed under the MIT License.
"""
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    print("README file generated successfully.")  # Debugging line


# Main function to execute the analysis
def main(input_file, output_dir):
    # Load the dataset
    df = pd.read_csv(input_file)

    # Analyze the data
    summary_stats, missing_values, corr_matrix = analyze_data(df)

    # Detect outliers
    outliers = detect_outliers(df)

    # Visualize the data
    visualize_data(corr_matrix, outliers, df, output_dir)

    # Generate README
    create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir)

    print("Data analysis and visualization completed.")  # Debugging line


# Example usage
if __name__ == "__main__":
    input_file = 'your_dataset.csv'  # Replace with your dataset path
    output_dir = 'output'  # Replace with your desired output directory
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    main(input_file, output_dir)
