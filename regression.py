# %%
# Imports
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# %%
# Load data
def get_file_path(file_name):
    path = f"{file_name}"
    if not os.path.exists(path):
        path = f"/your_code/{path}"
    return path


def set_random_seed(seed):
    # Sets random seed for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


X_df = pd.read_csv(
    get_file_path("Data/precomputed_node_embeddings.gz"), compression="gzip"
)
user_df = pd.read_csv(get_file_path("Data/precomputed_scores.gz"), compression="gzip")

print("This is a node embedding matrix of each node in the graph")
print(X_df.head())

# %%
# Train-test split
Y = pd.Series(user_df["broker_score"])
X_train, X_test, Y_train, Y_test = train_test_split(
    X_df, Y, train_size=0.9, shuffle=True, random_state=0
)

Y_class = pd.cut(
    Y,
    bins=[-float("inf"), 0, 25, 100, 500, 10000, float("inf")],
    labels=[0, 1, 2, 3, 4, 5],
)

Y_train_class = pd.cut(
    Y_train,
    bins=[-float("inf"), 0, 25, 100, 500, 10000, float("inf")],
    labels=[0, 1, 2, 3, 4, 5],
).astype(int)
Y_test_class = pd.cut(
    Y_test,
    bins=[-float("inf"), 0, 25, 100, 500, 10000, float("inf")],
    labels=[0, 1, 2, 3, 4, 5],
).astype(int)

# Balance the training data to have equal numbers of 10,000+ and less than 10,000 samples
high_value_indices = Y_train[Y_train >= 10000].index
medium_value_indices = Y_train[(Y_train >= 500) & (Y_train < 10000)].index
low_value_indices = Y_train[Y_train < 10000].index

# Sample from the low value indices to match the number of high value samples
low_value_sampled_indices = np.random.choice(
    low_value_indices, size=len(high_value_indices), replace=False
)
medium_value_sampled_indices = np.random.choice(
    medium_value_indices, size=len(high_value_indices), replace=False
)

# Combine the high value indices with the sampled low value indices
balanced_indices = np.concatenate([high_value_indices, medium_value_sampled_indices])

# Create balanced X_train and Y_train for binary classification
X_train_balanced = X_train.loc[balanced_indices]
Y_train_balanced = Y_train.loc[balanced_indices]

Y_train_binary_balanced = pd.cut(
    Y_train_balanced,
    bins=[-float("inf"), 10000, float("inf")],
    labels=[0, 1],
).astype(int)

Y_train_binary = pd.cut(
    Y_train,
    bins=[-float("inf"), 10000, float("inf")],
    labels=[0, 1],
).astype(int)
Y_test_binary = pd.cut(
    Y_test,
    bins=[-float("inf"), 10000, float("inf")],
    labels=[0, 1],
).astype(int)


# %%
# Define model training/evaluation functions
def train_eval(model, prediction_type="regression", **kwargs):
    set_random_seed(0)
    if prediction_type == "classification":
        model.fit(X_train, Y_train_class, **kwargs)
    elif prediction_type == "binary":
        model.fit(X_train, Y_train_binary, **kwargs)
    elif prediction_type == "balanced_binary":
        model.fit(X_train_balanced, Y_train_binary_balanced, **kwargs)
    elif prediction_type == "regression":
        model.fit(X_train, Y_train, **kwargs)
    scores = eval(model, prediction_type)
    return model, scores


def eval(model, prediction_type):
    scores = {"r2": None, "f1": None}
    Y_test_pred = model.predict(X_test)

    if prediction_type == "regression":
        r2 = r2_score(Y_test, Y_test_pred)
        scores["r2"] = r2
        print(f"    R2 score on test: {r2}")
        for i in range(10):
            print(f"        Actual: {Y_test.iloc[i]}, Predicted: {Y_test_pred[i]}")
    elif prediction_type == "classification":
        Y_test_pred_class = Y_test_pred
        f1 = f1_score(Y_test_class, Y_test_pred_class, average="weighted")
        scores["f1"] = f1
        print(f"    F1 score on test: {f1}")
        for i in range(10):
            print(
                f"        Actual: {Y_test_class.iloc[i]}, Predicted: {Y_test_pred_class[i]}"
            )
    elif prediction_type in ["binary", "balanced_binary"]:
        Y_test_pred_binary = Y_test_pred
        f1 = f1_score(Y_test_binary, Y_test_pred_binary, average="weighted")
        scores["f1"] = f1
        print(f"    F1 score on test: {f1}")
        for i in range(10):
            print(
                f"        Actual: {Y_test_binary.iloc[i]}, Predicted: {Y_test_pred_binary[i]}"
            )
    return scores


class nn_model(tf.keras.Sequential):
    # Wrapper class for tf.keras.Sequential that returns flattened predictions (in sklearn format)
    def predict(self, X):
        predictions = super().predict(X)
        if len(predictions[0]) == 1:
            return predictions.flatten()
        else:
            return np.array([np.argmax(x) for x in predictions])


class ensemble_model:
    # Model that uses the most common prediction among a set of models
    def __init__(self, models):
        self.models = models

    def fit(self, *args):
        pass

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        majority_vote = mode(predictions, axis=0)
        return np.array(majority_vote[0])


def create_tf_model(input_shape, layer_sizes, prediction_type):
    layers = []
    for i, size in enumerate(layer_sizes):
        if i == 0:
            layers.append(
                tf.keras.layers.Dense(
                    size, activation="relu", input_shape=(input_shape,)
                )
            )
        elif i == len(layer_sizes) - 1:
            if prediction_type in ["binary", "balanced_binary", "classification"]:
                layers.append(tf.keras.layers.Dense(size, activation="softmax"))
            elif prediction_type == "regression":
                layers.append(tf.keras.layers.Dense(size, activation="relu"))
        else:
            if prediction_type in ["binary", "balanced_binary", "classification"]:
                layers.append(tf.keras.layers.Dense(size, activation="relu"))
            elif prediction_type == "regression":
                layers.append(tf.keras.layers.Dense(size))
    model = nn_model(layers)
    if prediction_type in ["binary", "balanced_binary", "classification"]:
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# %%
classification_mode = "binary"  # "classification", "balanced_binary", or "binary"
# %%
# Linear regression
print("Linear Regression")
linear_regression = train_eval(
    linear_model.LinearRegression(), prediction_type="regression"
)

# train_eval(linear_model.LogisticRegressionCV(cv=2, max_iter=3), X_df, Y, train_size=0.2)
#   Takes a long time to run, tries to allocate too much memory (~50 GB) if attempted on full training set

# %%
# Random Forest Classifier
print("Random Forest Classifier")
rf_classifier, rf_classifier_score = train_eval(
    RandomForestClassifier(n_estimators=25, random_state=0),
    prediction_type=classification_mode,
)
# %%
# K-Nearest Neighbors Classifier
print("K-Nearest Neighbors Classifier")
knn_classifier, knn_classifier_score = train_eval(
    KNeighborsClassifier(n_neighbors=5), prediction_type=classification_mode
)
# %%
# Sequential Neural Network Regression
print("Sequential Neural Network (regression) [64, 32, 1]")
snn, snn_score = train_eval(
    create_tf_model(
        X_df.shape[1], layer_sizes=[64, 32, 1], prediction_type="regression"
    ),
    validation_data=(X_test, Y_test),
    epochs=100,
    batch_size=16384,
    verbose=0,
    prediction_type="regression",
)
# Batch size greatly increases training speed - lower if memory issues arise

# %%
# Sequential Neural Network Classifiers
if classification_mode in ["binary", "balanced_binary"]:
    print("Sequential Neural Network (classifier) [64, 48, 2]")
    snn_classify_1, snn_classify_score_1 = train_eval(
        create_tf_model(
            X_df.shape[1],
            layer_sizes=[64, 48, 48, 2],
            # Final layer size must match number of classes - largest output weight is chosen
            prediction_type=classification_mode,
        ),
        validation_data=(X_test, Y_test_binary),
        epochs=100,
        batch_size=16384,
        verbose=0,
        prediction_type=classification_mode,
    )
else:
    print("Sequential Neural Network (classifier) [64, 48, 6]")
    snn_classify_1, snn_classify_score_1 = train_eval(
        create_tf_model(
            X_df.shape[1],
            layer_sizes=[64, 48, 48, 6],
            # Final layer size must match number of classes - largest output weight is chosen
            prediction_type=classification_mode,
        ),
        validation_data=(X_test, Y_test_class),
        epochs=100,
        batch_size=16384,
        verbose=0,
        prediction_type=classification_mode,
    )
# %%
# Sequential Neural Network Classifiers 2
if classification_mode in ["binary", "balanced_binary"]:
    print("Sequential Neural Network (binary classifier) [144, 144, 144, 144, 144, 2]")
    snn_classify_2, snn_classify_score_2 = train_eval(
        create_tf_model(
            X_df.shape[1],
            layer_sizes=[144, 144, 144, 144, 144, 2],
            prediction_type=classification_mode,
        ),
        validation_data=(X_test, Y_test_binary),
        epochs=100,
        batch_size=16384,
        verbose=0,
        prediction_type="binary",
    )
# %%
# Sequential Neural Network Classifiers 3
if classification_mode in ["binary", "balanced_binary"]:
    print(
        "Sequential Neural Network (balanced binary classifier) [144, 144, 144, 144, 144, 2]"
    )
    snn_classify_3, snn_classify_score_3 = train_eval(
        create_tf_model(
            X_df.shape[1],
            layer_sizes=[144, 144, 144, 144, 144, 2],
            prediction_type=classification_mode,
        ),
        validation_data=(X_test, Y_test_binary),
        epochs=100,
        batch_size=16384,
        verbose=0,
        prediction_type="balanced_binary",
    )
else:
    print("Sequential Neural Network (classifier) [144, 144, 144, 144, 144, 6]")
    snn_classify_2, snn_classify_score_2 = train_eval(
        create_tf_model(
            X_df.shape[1],
            layer_sizes=[144, 144, 144, 144, 144, 6],
            prediction_type=classification_mode,
        ),
        validation_data=(X_test, Y_test_class),
        epochs=100,
        batch_size=16384,
        verbose=0,
        prediction_type=classification_mode,
    )

# %%
# Ensemble Classifier
print(
    "Ensemble Classifier: Sequential Neural Network, Random Forest, K-Nearest Neighbors"
)
ensemble_classify, ensemble_classify_score = train_eval(
    ensemble_model([snn_classify_2, rf_classifier, knn_classifier]),
    prediction_type=classification_mode,
)
ensemble_predictions = ensemble_classify.predict(X_test)

# %%
# Visualize broker score distributions
plt.figure(figsize=(10, 6))
plt.hist(Y, bins=50, color="blue", alpha=0.7, log=True)
plt.title("Log Distribution of Broker Scores")
plt.xlabel("Broker Score")
plt.ylabel("Log Frequency")
plt.grid(True)
plt.savefig(get_file_path(f"Data/Log Distribution of Broker Scores.png"))
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(
    Y_class, bins=range(7), align="left", color="blue", alpha=0.7, rwidth=0.8, log=True
)
plt.title("Log Distribution of Broker Score Classes")
plt.xlabel("Broker Class")
plt.ylabel("Log Frequency")
plt.xticks(range(6), ["0", "1-25", "26-99", "100-499", "500-9,999", "10,000+"])
plt.grid(True)
plt.savefig(get_file_path(f"Data/Log Distribution of Broker Score Classes.png"))
plt.show()


# %%
# Visualization
def visualize(X, Y_class, visualizer, title: str, dimensions=2, data_size=None):
    X_visualized, X_hidden, Y_visualized, Y_hidden = train_test_split(
        X, Y_class, train_size=data_size, random_state=0
    )
    X_transformed = visualizer.fit_transform(X_visualized)
    if dimensions == 2:
        plt.title(title)
        scatter = plt.scatter(
            X_transformed[:, 0],
            X_transformed[:, 1],
            c=Y_visualized,
            s=3,
            cmap=plt.cm.prism,
        )
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=["0", "1-25", "26-99", "100-499", "500-9,999", "10,000+"],
        )
        plt.xlabel("Reduced Dimension 1")
        plt.ylabel("Reduced Dimension 2")
        scatter.set_sizes([0.1])

    elif dimensions == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d", title=title)
        scatter = ax.scatter(
            X_transformed[:, 0],
            X_transformed[:, 1],
            X_transformed[:, 2],
            c=Y_visualized,
            s=3,
            cmap=plt.cm.prism,
        )
        ax.legend(
            handles=scatter.legend_elements()[0],
            labels=["0", "1-25", "26-99", "100-499", "500-9,999", "10,000+"],
        )
        ax.set_xlabel("Reduced Dimension 1")
        ax.set_ylabel("Reduced Dimension 2")
        ax.set_zlabel("Reduced Dimension 3")
        ax.set_box_aspect(None, zoom=0.9)
        scatter.set_sizes([0.1])

    plt.savefig(get_file_path(f"Data/{title}.png"))
    plt.show()


allow_visualize = input(
    "Type visualize to re-compute visualizations, or press enter to show pre-computed visualizations: "
)
if allow_visualize.lower() == "visualize":
    visualize(X_df, Y_class, PCA(n_components=2), "Broker Score (2D PCA)", dimensions=2)
    visualize(X_df, Y_class, PCA(n_components=3), "Broker Score (3D PCA)", dimensions=3)
    visualize(
        X_df,
        Y_class,
        TSNE(n_components=2),
        "Broker Score (2D t-SNE)",
        dimensions=2,
        data_size=0.05,
    )
    visualize(
        X_df,
        Y_class,
        TSNE(n_components=3),
        "Broker Score (3D t-SNE)",
        dimensions=3,
        data_size=0.05,
    )
    visualize(
        X_test,
        ensemble_predictions,
        PCA(n_components=2),
        "Broker Score Ensemble Predictions (2D PCA)",
        dimensions=2,
    )
    visualize(
        X_test,
        Y_test_class,
        PCA(n_components=2),
        "Broker Score Actual (2D PCA)",
        dimensions=2,
    )

else:
    for image in [
        "Broker Score (2D PCA)",
        "Broker Score (3D PCA)",
        "Broker Score (2D t-SNE)",
        "Broker Score (3D t-SNE)",
        "Broker Score Ensemble Predictions (2D PCA)",
        "Broker Score Actual (2D PCA)",
    ]:
        plt.imshow(mpimg.imread(get_file_path(f"Data/{image}.png")))
        plt.show()

# %%
# Binary classification
if classification_mode in ["binary", "balanced_binary"]:
    binary_ensemble_predictions = ensemble_predictions
else:
    binary_ensemble_predictions = np.array(
        [0 if 0 <= pred <= 4 else 1 for pred in ensemble_predictions]
    )
binary_actual = np.array([0 if 0 <= act <= 4 else 1 for act in Y_test_class])
binary_f1 = f1_score(binary_actual, binary_ensemble_predictions, average="weighted")
print(f"Ensemble binary F1 score (distinguish 10,000+ brokers): {binary_f1}")
conf_matrix = confusion_matrix(binary_actual, binary_ensemble_predictions)
print("Confusion Matrix:\n")
print("[[TN  FP]")
print(" [FN  TP]]\n")
print(conf_matrix)

# %%
# Multi-NN binary classification
if classification_mode in ["binary", "balanced_binary"]:
    # snn_classify_2 is unbalanced, snn_classify_3 is balanced
    # Binary classification 2
    if classification_mode in ["binary", "balanced_binary"]:
        binary_nn_predictions = snn_classify_2.predict(X_test)
    else:
        binary_nn_predictions = np.array(
            [0 if 0 <= pred <= 4 else 1 for pred in snn_classify_2.predict(X_test)]
        )
    binary_actual = np.array([0 if 0 <= act <= 4 else 1 for act in Y_test_class])
    binary_f1 = f1_score(binary_actual, binary_nn_predictions, average="weighted")
    print(
        f"Unbalanced Neural network F1 score (distinguish 10,000+ brokers): {binary_f1}"
    )
    conf_matrix = confusion_matrix(binary_actual, binary_nn_predictions)
    print("Confusion Matrix:\n")
    print("[[TN  FP]")
    print(" [FN  TP]]\n")
    print(conf_matrix)

    if classification_mode in ["binary", "balanced_binary"]:
        balanced_binary_nn_predictions = snn_classify_3.predict(X_test)
    else:
        balanced_binary_nn_predictions = np.array(
            [0 if 0 <= pred <= 4 else 1 for pred in snn_classify_3.predict(X_test)]
        )
    binary_actual = np.array([0 if 0 <= act <= 4 else 1 for act in Y_test_class])
    binary_f1 = f1_score(
        binary_actual, balanced_binary_nn_predictions, average="weighted"
    )
    print(
        f"Balanced Neural network F1 score (distinguish 10,000+ brokers): {binary_f1}"
    )
    conf_matrix = confusion_matrix(binary_actual, balanced_binary_nn_predictions)
    print("Confusion Matrix:\n")
    print("[[TN  FP]")
    print(" [FN  TP]]\n")
    print(conf_matrix)

    cooperative_predictions = np.array(
        [
            1 if pred1 == 1 and pred2 == 1 else 0
            for pred1, pred2 in zip(
                binary_nn_predictions, balanced_binary_nn_predictions
            )
        ]
    )
    binary_f1 = f1_score(binary_actual, cooperative_predictions, average="weighted")
    print(
        f"Cooperative Neural network F1 score (distinguish 10,000+ brokers): {binary_f1}"
    )
    conf_matrix = confusion_matrix(binary_actual, cooperative_predictions)
    print("Confusion Matrix:\n")
    print("[[TN  FP]")
    print(" [FN  TP]]\n")
    print(conf_matrix)

# We want to maximize true positives / false negatives to avoid missing any high-value brokers
# Alternatively, maximize true positives / false positives to avoid wasting time on low-value brokers

# Maybe save models as files for later use w/o retraining
# Add specialized models that only identify 10,000+ and below 10,000 - find most influential brokers
#   Look into neural network configuration for rare class identification

# Show main results in presentation
# Go through problem definition, etc.
# Try larger node embeddings - 64-128+
# To replicate paper, try to identify top p% of brokers, with p being 5 or 10%
# The paper got rather low F1 scores of 0.4-0.6, depite having larger datasets
# Try different KNN parameters
# Try comparing pure binary and multi-class -> binary classification performance
# Note: balanced binary classification had very high recall but low precision - many false positives
#   If balanced says positive and unbalanced says positive, it is likely positive
#   If balanced says positive and unbalanced says negative, it is likely negative
#   If balanced says negative and unbalanced says positive, it is likely negative
#   If balanced says negative and unbalanced says negative, it is likely negative
# Possibly try a cooperative ensemble with binary/balanced versions of all 3 classifier types - only positive if they all vote positive
# Additionally, the unbalanced-only ensemble should vote positive only if all agree, rather than majority
# %%
