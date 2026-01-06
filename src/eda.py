import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir='reports'):
    """
    Generates basic Exploratory Data Analysis plots and saves them.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print("Generating EDA reports...")
    
    # 1. Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='machine_failure', data=df)
    plt.title('Target Distribution: Machine Failure')
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
    plt.close()
    
    # 2. Correlation Matrix (Numerical only)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    print(f"EDA complete. Plots saved to '{output_dir}/'")
