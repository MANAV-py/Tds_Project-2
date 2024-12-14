import os
import sys
import openai
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the token directly from the environment variable
token = os.environ.get("AIPROXY_TOKEN")
if not token:
    raise EnvironmentError("AIPROXY_TOKEN not found in environment variables. Please check your .env file.")

# Configure OpenAI with the token and set the correct base URL for AI Proxy
openai.api_key = token
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

def analyze_csv(filename, output_dir="."):
    """Load and analyze a CSV file, performing analysis and saving outputs to a specified directory."""
    try:
        # Load dataset with flexible encoding
        data = pd.read_csv(filename, encoding='ISO-8859-1')
        print(f"Dataset loaded successfully: {filename}")
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Perform generic analysis
    print("Performing generic analysis...")
    print(data.info())
    print("\nSummary statistics:")
    print(data.describe())

    # Count missing values
    missing_values = data.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values)

    # Filter out numeric columns for analysis
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Handle missing values in numeric data (drop rows with NaN values)
    numeric_data_cleaned = numeric_data.dropna()

    # Correlation Heatmap
    try:
        correlation_matrix = numeric_data_cleaned.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        heatmap_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_correlation_heatmap.png")
        plt.savefig(heatmap_path)
        print(f"Correlation heatmap saved as {heatmap_path}")
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")

    # Pairplot
    try:
        pairplot_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_pairplot.png")
        sns.pairplot(numeric_data_cleaned)
        plt.savefig(pairplot_path)
        print(f"Pairplot saved as {pairplot_path}")
    except Exception as e:
        print(f"Error creating pairplot: {e}")

    # Regression Plot
    try:
        if 'Life Ladder' in numeric_data_cleaned.columns and 'Log GDP per capita' in numeric_data_cleaned.columns:
            plt.figure(figsize=(10, 6))
            sns.regplot(x='Log GDP per capita', y='Life Ladder', data=numeric_data_cleaned, line_kws={"color": "red"})
            regression_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_regression_plot.png")
            plt.savefig(regression_path)
            print(f"Regression plot saved as {regression_path}")
    except Exception as e:
        print(f"Error creating regression plot: {e}")

    # Histograms
    histogram_paths = []  # List to store histogram paths
    try:
        for column in numeric_data_cleaned.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(numeric_data_cleaned[column], kde=True)
            hist_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{column}_histogram.png")
            plt.savefig(hist_path)
            histogram_paths.append(hist_path)  # Store the path
            print(f"Histogram for {column} saved as {hist_path}")
    except Exception as e:
        print(f"Error creating histograms: {e}")

    # Outlier Detection using IQR
    try:
        Q1 = numeric_data_cleaned.quantile(0.25)
        Q3 = numeric_data_cleaned.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_data_cleaned < (Q1 - 1.5 * IQR)) | (numeric_data_cleaned > (Q3 + 1.5 * IQR))).sum()
        print("\nOutliers per column:")
        print(outliers)
    except Exception as e:
        print(f"Error detecting outliers: {e}")

    # Clustering Analysis
    clustered_data_path = None
    
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data_cleaned)
        inertias = []
        silhouettes = []
        optimal_k = 3
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(scaled_data, cluster_labels))
        optimal_k = silhouettes.index(max(silhouettes)) + 2
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        data['Cluster'] = pd.Series(clusters, index=numeric_data_cleaned.index)
        clustered_data_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_clustered.csv")
        data.to_csv(clustered_data_path, index=False)
        print(f"Clustered data saved as {clustered_data_path}")
    except Exception as e:
        print(f"Error performing clustering analysis: {e}")

    # Generate insights using LLM
    print("Generating insights using LLM...")
    context = f"Column names: {list(data.columns)}\nSummary: {data.describe()}\nMissing Values: {missing_values}\nOutliers: {outliers}"
    insights = generate_readme(data, context, output_dir)

    # Save the insights and visualization to README.md
    try:
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as file:
            file.write("# Dataset Analysis Report\n\n")
            file.write("## Insights from LLM\n\n")
            file.write(insights)
            file.write("\n\n## Visualizations\n")
            file.write(f"- [Correlation Heatmap](./{os.path.basename(heatmap_path)})\n")
            file.write(f"- [Pairplot](./{os.path.basename(pairplot_path)})\n")
            if 'Life Ladder' in numeric_data_cleaned.columns and 'Log GDP per capita' in numeric_data_cleaned.columns:
                file.write(f"- [Regression Plot](./{os.path.basename(regression_path)})\n")
            for hist_path in histogram_paths:  # Iterate over histogram paths
                file.write(f"- [Histogram for {os.path.basename(hist_path)}](./{os.path.basename(hist_path)})\n")
            if clustered_data_path:
                file.write(f"- [Clustered Data CSV](./{os.path.basename(clustered_data_path)})\n")
            else:
                file.write("Clustering analysis failed.")
        print(f"Report saved as {readme_path}")
    except Exception as e:
        print(f"Error saving report: {e}")

def generate_readme(data, context, output_dir):
    """
    Generate README.md content using GPT-4o-Mini via AI Proxy.
    """
    prompt = f"""
    You are an expert data analyst. Analyze the following dataset and its visualizations, then write a detailed README.md file:

    ### Dataset Description
    {data.info()}

    ### Summary Statistics
    {data.describe()}

    ### Missing Values
    {context}

    ### Outlier Analysis
    {context}

    ### Visualizations
    - Correlation heatmap (saved as correlation_heatmap.png in {output_dir}).
    - Pairplot (saved as pairplot.png in {output_dir}).
    - Regression plot (saved as regression_plot.png in {output_dir}).
    - Histograms for each numeric column (saved in {output_dir}).
    - KMeans Clustering (clustered data saved as [clustered_data.csv]({output_dir}/clustered_data.csv)).

    Write the README.md in Markdown format. Include headers, clear descriptions, analysis steps, key findings, and links to visualizations.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an expert data analyst."},
                      {"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating README.md: {e}")
        return "Failed to generate README.md content."

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_directory = "."  # Default to the current working directory
    analyze_csv(csv_file, output_directory)