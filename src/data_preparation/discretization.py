import pandas as pd
import os
import torch


def discretize_gene_expression_using_z_score(df: pd.DataFrame, z_threshold=2):
    """
    Discretizes gene expression levels into 'up', 'down', 'no_change' based on z-scores.

    :param df: 2D pandas DataFrame with gene expression (rows=samples, columns=genes)
    :param z_threshold: Z-score threshold to define up/down regulation
    :return: Discretized DataFrame with string values: 'up', 'down', 'no_change'
    """
    # Compute z-scores across samples for each gene
    z_scores = (df - df.mean()) / df.std(ddof=0)

    # Apply discretization rules
    df_discrete = pd.DataFrame(index=df.index, columns=df.columns)
    df_discrete[z_scores > z_threshold] = 'up'
    df_discrete[z_scores < -z_threshold] = 'down'
    df_discrete[(z_scores <= z_threshold) & (z_scores >= -z_threshold)] = 'no_change'

    return df_discrete


def discretize_normalized_gene_expression(df):
    """
    Discretize normalized gene expression DataFrame into 'low', 'normal', and 'high' string labels.

    Parameters:
        df (pd.DataFrame): Gene expression values (standardized: mean=0, std=1)

    Returns:
        pd.DataFrame: Same shape as input, with 'low', 'normal', or 'high' in each cell
    """

    # Use vectorized operations with DataFrame.applymap for efficiency
    def categorize(value):
        if value <= -1:
            return 'low'
        elif value >= 1:
            return 'high'
        else:
            return 'normal'

    return df.applymap(categorize)


directory = "../../data/raw"
target_directory = "../../data/discrete"

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        data = torch.load(filepath, weights_only=False)

        X_expression = data['X_expression']

        table = pd.DataFrame(X_expression.numpy(), columns=[f'gene_{i + 1}' for i in range(X_expression.shape[1])])

        table = discretize_normalized_gene_expression(table)
        table.to_csv(target_directory + "/" + filename.replace('.pt', '') + ".csv", index=False)
