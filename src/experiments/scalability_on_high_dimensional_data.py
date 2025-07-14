import time
import csv
import warnings
from datetime import datetime

from mlxtend.frequent_patterns import fpgrowth, association_rules, hmine
from aerial import model, rule_extraction, rule_quality
from src.arm_ae.armae import ARMAE

import pandas as pd
from pyECLAT import ECLAT

# todo: resolve the warnings
warnings.filterwarnings("ignore")


def load_gene_expression_data(dataset_name):
    file_name = "./data/discrete/" + dataset_name + ".csv"
    gene_expression = pd.read_csv(file_name)

    return gene_expression


def run_fpgrowth(dataset, min_support=0.5, min_confidence=0.8, antecedents=2):
    start = time.time()
    frequent_patterns = fpgrowth(dataset, min_support=min_support, use_colnames=True, max_len=antecedents + 1)
    rules = association_rules(frequent_patterns, metric="confidence", min_threshold=min_confidence)
    return rules, (time.time() - start)


def run_hmine(dataset, min_support=0.5, min_confidence=0.8, antecedents=2):
    start = time.time()
    frequent_patterns = hmine(dataset, min_support=min_support, use_colnames=True, max_len=antecedents + 1)
    rules = association_rules(frequent_patterns, metric="confidence", min_threshold=min_confidence)
    return rules, (time.time() - start)


def run_eclat(dataset, min_support=0.5, min_confidence=0.8, antecedents=2):
    """
    to run ECLAT:
    transactions = subset.apply(
                    lambda row: [str(f"{col}_{row[col]}") for col in subset.columns],
                    axis=1
                )
                max_len = transactions.map(len).max()
                transaction_df = pd.DataFrame(transactions.tolist(), columns=list(range(max_len)))
                rules, exec_time = run_eclat(transaction_df, min_support=0.5, min_confidence=0.8, antecedents=2)
    """
    start = time.time()
    eclat_instance = ECLAT(data=dataset)
    indexes, support = eclat_instance.fit(min_support=min_support, min_combination=1, max_combination=antecedents + 1,
                                          verbose=False)
    total = time.time() - start
    frequent_itemsets = pd.DataFrame({
        'itemsets': [frozenset(itemset.split(" & ")) for itemset in support.keys()],
        'support': list(support.values())
    })
    start = time.time()
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    total += time.time() - start
    return rules, total


def run_aerial(dataset, ant_sim=0.5, cons_sim=0.8, antecedents=2, epochs=2):
    start = time.time()
    trained_autoencoder = model.train(dataset, epochs=epochs)
    # extract association rules from the autoencoder
    association_rules = rule_extraction.generate_rules(trained_autoencoder, ant_similarity=ant_sim,
                                                       cons_similarity=cons_sim, max_antecedents=antecedents)
    return association_rules, (time.time() - start), trained_autoencoder


gene_counts = [10, 20]
dataset_names = ["cell_cancertype_Melanoma_lnic50"]
print("INFO: Running the scalability experiments on 'cell_cancertype_Melanoma_lnic50' "
      "dataset with up to 20 genes (columns).")
print("INFO: To run the experiments on other datasets, please update 'dataset_names' objects in line 74, "
      "or with a higher number of genes, please update line 73.")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"scalability_experiments_{timestamp}.csv"
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow([
        "Dataset", "Algorithm", "Gene Count", "One-Hot Gene Count",
        "Rule Count", "Execution Time", "Average Support"
    ])

    for dataset_name in dataset_names:
        dataset = load_gene_expression_data(dataset_name)
        print("Dataset:", dataset_name)

        # Write a row with only the dataset name
        writer.writerow([dataset_name])

        for genes in gene_counts:
            print("Genes:", genes)
            subset = dataset.iloc[:, :genes]
            one_hot_encoded = pd.get_dummies(subset)
            # AERIAL
            aerial_rules, aerial_time, trained_ae = run_aerial(
                subset, ant_sim=0.5, cons_sim=0.8, antecedents=2, epochs=10
            )
            aerial_rules = rule_quality.calculate_basic_rule_stats(aerial_rules, trained_ae.input_vectors)
            if aerial_rules and len(aerial_rules) > 0:
                aerial_avg_support = sum(d["support"] for d in aerial_rules) / len(aerial_rules)
            else:
                aerial_avg_support = 0.5
            writer.writerow([
                "", "aerial", genes, one_hot_encoded.shape[1],
                len(aerial_rules) if aerial_rules else 0, aerial_time, aerial_avg_support
            ])
            print(
                f"Aerial | Genes: {genes} | Features (one-hot encoded): {one_hot_encoded.shape[1]} | Rules: "
                f"{len(aerial_rules) if aerial_rules is not None else 0} | Time: {aerial_time:.2f}s | "
                f"Avg Support: {aerial_avg_support:.2f}")

            arm_ae = ARMAE(len(one_hot_encoded.loc[0]), maxEpoch=10, batchSize=2, likeness=0.5, IM=["support"])
            dataLoader = arm_ae.dataPreprocessing(one_hot_encoded)
            arm_ae.train(dataLoader)
            # numberOfRules per consequent is adjusted to approximate aerial_plus+
            arm_ae.generateRules(one_hot_encoded,
                                 numberOfRules=max(
                                     int(len(aerial_rules) / one_hot_encoded.shape[1]), 2),
                                 nbAntecedent=2)
            arm_ae_exec_time = arm_ae.arm_ae_training_time + arm_ae.exec_time
            if arm_ae.results and len(arm_ae.results) > 0:
                arm_ae_avg_support = sum(d["support"] for d in arm_ae.results) / len(arm_ae.results)
            else:
                arm_ae_avg_support = 0.5
            writer.writerow([
                "", "arm-ae", genes, one_hot_encoded.shape[1],
                len(arm_ae.results) if arm_ae.results else 0, arm_ae_exec_time, arm_ae_avg_support
            ])
            print(
                f"ARM-AE | Genes: {genes} | Features (one-hot encoded): {one_hot_encoded.shape[1]} | Rules: {len(arm_ae.results) if arm_ae.results else 0} | Time: {arm_ae_exec_time:.2f}s | Avg Support: {arm_ae_avg_support:.2f}")

            # FP-Growth
            fpg_rules, fpg_time = run_fpgrowth(
                one_hot_encoded, min_support=0.5,
                min_confidence=0.8, antecedents=2
            )
            fpg_support = fpg_rules["support"].mean()
            writer.writerow(["", "fpgrowth", genes, one_hot_encoded.shape[1], len(fpg_rules), fpg_time, fpg_support])
            print(
                f"FP-Growth | Genes: {genes} | Features (one-hot encoded): {one_hot_encoded.shape[1]} | "
                f"Rules: {len(fpg_rules) if fpg_rules is not None else 0} | "
                f"Time: {fpg_time:.2f}s | Avg Support: {fpg_support:.2f}")

            # ECLAT
            transactions = dataset.apply(
                lambda row: [str(f"{col}_{row[col]}") for col in subset.columns],
                axis=1
            )
            max_len = transactions.map(len).max()
            transaction_df = pd.DataFrame(transactions.tolist(), columns=list(range(max_len)))
            eclat_rules, eclat_time = run_eclat(transaction_df, min_support=0.5, min_confidence=0.8, antecedents=2)
            eclat_mean_support = eclat_rules["support"].mean()
            writer.writerow([
                "", "eclat", genes, one_hot_encoded.shape[1],
                len(eclat_rules), eclat_time, eclat_mean_support
            ])
            print(
                f"ECLAT | Genes: {genes} | Features (one-hot encoded): {one_hot_encoded.shape[1]} | "
                f"Rules: {len(eclat_rules) if eclat_rules is not None else 0} | "
                f"Time: {eclat_time:.2f}s | Avg Support: {eclat_mean_support:.2f}")
