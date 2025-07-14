import time
from collections import defaultdict

import aerial
import pandas as pd
from aerial import model, rule_extraction, rule_quality
from src.fine_tuned_aerial import model_double_loss_pretrained
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding


def load_gene_expression_data(dataset):
    file_name = "./data/discrete/" + dataset + ".csv"
    gene_expression = pd.read_csv(file_name)

    X = gene_expression.iloc[:, :100]
    y = gene_expression.iloc[:, 100:101]
    return X, y


def get_average_rule_quality_stats(rule_quality_stats):
    agg = defaultdict(float)
    n = len(rule_quality_stats)

    for d in rule_quality_stats:
        for k, v in d.items():
            agg[k] += v

    return {k: round(float(v / n), 3) for k, v in agg.items()}


if __name__ == '__main__':
    NUMBER_OF_RUNS = 50
    EPOCHS = [25]
    BATCH_SIZE = 2
    LAYER_DIMS = [50, 10]

    dataset_names = ["cell_cancertype_Melanoma_lnic50"]
    print("INFO: Running the Aerial+DL experiments on 'cell_cancertype_Melanoma_lnic50' dataset "
          "with 100 genes (columns).")
    print("INFO: To run the experiments on other datasets, please update 'dataset_names' objects in line 39.")
    for USE_TABPFN_EMBEDDINGS in [True, False]:
        for dataset in dataset_names:
            print("Dataset:", dataset)
            X, y = load_gene_expression_data(dataset)
            X = pd.DataFrame(X).reset_index(drop=True)
            y = pd.DataFrame(y).reset_index(drop=True)
            X = X.dropna()
            y = y.loc[X.index]
            aerial_train = pd.concat([X, y], axis=1)

            for epoch in EPOCHS:
                rule_quality_statistics = []
                exec_times = []

                print("Epochs:", epoch)
                for i in range(NUMBER_OF_RUNS):
                    print(f"Number of Runs [{i + 1}/{NUMBER_OF_RUNS}]", end="\n", flush=True)
                    X = pd.DataFrame(X).reset_index(drop=True)
                    y = pd.DataFrame(y).reset_index(drop=True)
                    X = X.dropna()
                    y = y.loc[X.index]

                    aerial_train = pd.concat([X, y], axis=1)

                    start = time.time()
                    if USE_TABPFN_EMBEDDINGS:
                        clf = TabPFNClassifier(n_estimators=1, ignore_pretraining_limits=True)
                        embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf)
                        tabpfn_embeddings = embedding_extractor.get_embeddings(X.to_numpy(), y.to_numpy().flatten(),
                                                                               X.to_numpy(),
                                                                               data_source="train").squeeze(0)
                        trained_autoencoder = model_double_loss_pretrained.train(aerial_train, tabpfn_embeddings,
                                                                                 epochs=epoch, batch_size=BATCH_SIZE,
                                                                                 layer_dims=LAYER_DIMS)
                    else:
                        trained_autoencoder = model.train(aerial_train, epochs=epoch, batch_size=BATCH_SIZE,
                                                          layer_dims=LAYER_DIMS)

                    features_of_interest = [{feature.split("__")[0]: feature.split("__")[1]} for feature in
                                            trained_autoencoder.feature_values if "normal" not in feature]
                    # extract association rules from the autoencoder
                    association_rules = rule_extraction.generate_rules(trained_autoencoder,
                                                                       features_of_interest=features_of_interest,
                                                                       target_classes=features_of_interest,
                                                                       ant_similarity=0.5)
                    exec_times.append(time.time() - start)

                    # calculate rule quality statistics (support, confidence, zhangs metric) for each rule
                    if len(association_rules) > 0:
                        stats, association_rules = rule_quality.calculate_rule_stats(association_rules,
                                                                                     trained_autoencoder.input_vectors,
                                                                                     max_workers=8)
                        rule_quality_statistics.append(stats)
                        print(stats)

                print(get_average_rule_quality_stats(rule_quality_statistics), (sum(exec_times) / len(exec_times)))
