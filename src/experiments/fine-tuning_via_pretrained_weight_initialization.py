import time
from collections import defaultdict

import aerial
import pandas as pd
import torch
from aerial import model, rule_extraction, rule_quality
from aerial.model import AutoEncoder
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding


def load_gene_expression_data(dataset):
    file_name = "./data/discrete/" + dataset + ".csv"
    gene_expression = pd.read_csv(file_name)

    X = gene_expression.iloc[:, :100]
    y = gene_expression.iloc[:, 100:101]
    return X, y


def get_tabpfn_initialized_autoencoder(X_train, y_train, aerial_train, layer_dims: list):
    # get TabPFN embeddings
    clf = TabPFNClassifier(n_estimators=1)
    # todo: the example in TabPFNEmbedding might be overdoing for our case: https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/embedding/tabpfn_embedding.py
    # The example splits the data into train and test, and then uses k-fold cross-validation on the training to generate
    # the embeddings. We can just skip the train-test split as our goal is not to learn rules over the unseen part
    embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=10)
    tabpfn_embeddings = embedding_extractor.get_embeddings(X_train, y_train, X_train, data_source="train").squeeze(0)

    tabpfn_embeddings = torch.tensor(tabpfn_embeddings, dtype=torch.float32)

    one_hot_encoded, feature_value_indices = model._one_hot_encoding_with_feature_tracking(aerial_train)
    X_1hot_tensor = torch.tensor(one_hot_encoded.to_numpy(), dtype=torch.float32)
    # X_1hot_tensor = F.layer_norm(X_1hot_tensor, X_1hot_tensor.shape[1:])  # normalize

    # Build projection encoder
    hidden_dim = layer_dims[0]
    projection_encoder = nn.Sequential(
        nn.Linear(X_1hot_tensor.shape[1], hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Dropout(p=0.1),
        nn.Linear(hidden_dim, tabpfn_embeddings.shape[1])
    )

    # Train projection encoder with hybrid cosine + MSE loss
    optimizer = torch.optim.Adam(projection_encoder.parameters(), lr=1e-3)
    dataset = TensorDataset(X_1hot_tensor, tabpfn_embeddings)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    projection_encoder.train()
    patience = 0
    best_loss = float('inf')
    for epoch in range(25):
        for x_1hot, emb in dataloader:
            optimizer.zero_grad()
            pred = projection_encoder(x_1hot)

            loss = 1 - F.cosine_similarity(pred, emb, dim=1).mean()

            if loss < best_loss - 1e-4:
                best_loss = loss
                patience = 0
            else:
                patience += 1
                if patience >= 20:
                    break

            loss.backward()
            optimizer.step()

        if patience >= 20:
            break  # breaks epoch loop, stops training fully

    # Initialize Autoencoder with projection weights
    autoencoder = AutoEncoder(
        input_dimension=one_hot_encoded.shape[1],
        feature_count=len(feature_value_indices),
        layer_dims=layer_dims
    )

    autoencoder.encoder[0].weight.data = projection_encoder[0].weight.data.clone()
    autoencoder.encoder[0].bias.data = projection_encoder[0].bias.data.clone()

    return autoencoder, one_hot_encoded, feature_value_indices


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
    USE_TABPFN_EMBEDDINGS = True
    LAYER_DIMS = [50, 10]

    dataset_names = ["cell_cancertype_Melanoma_lnic50"]
    print("INFO: Running the Aerial+WI experiments on 'cell_cancertype_Melanoma_lnic50' dataset "
          "with 100 genes (columns).")
    print("INFO: To run the experiments on other datasets, please update 'dataset_names' objects in line 124.")

    # "cell_cancertype_SmallCellLungCarcinoma_lnic50",
    # "cell_cancertype_BreastCarcinoma_lnic50",
    # "cell_cancertype_NonSmallCellLungCarcinoma_lnic50",
    # "cell_cancertype_Chondrosarcoma_lnic50"]
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

                start = time.time()
                if USE_TABPFN_EMBEDDINGS:
                    autoencoder, _, feature_value_indices = \
                        get_tabpfn_initialized_autoencoder(X.to_numpy(), y.to_numpy().flatten(), aerial_train,
                                                           LAYER_DIMS)
                    trained_autoencoder = model.train(aerial_train, autoencoder=autoencoder, epochs=epoch,
                                                      batch_size=BATCH_SIZE,
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

            print(get_average_rule_quality_stats(rule_quality_statistics), len(rule_quality_statistics), "/",
                  NUMBER_OF_RUNS)
            print("~ Execution time:", sum(exec_times) / len(exec_times))
