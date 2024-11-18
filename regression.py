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
    Y, bins=[-float("inf"), 0, 25, 100, 500, float("inf")], labels=[0, 1, 2, 3, 4]
)
Y_train_class = pd.cut(
    Y_train, bins=[-float("inf"), 0, 25, 100, 500, float("inf")], labels=[0, 1, 2, 3, 4]
).astype(int)
Y_test_class = pd.cut(
    Y_test, bins=[-float("inf"), 0, 25, 100, 500, float("inf")], labels=[0, 1, 2, 3, 4]
).astype(int)


# %%
# Visualize node embeddings
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
            labels=["0", "1-25", "26-99", "100-499", "500+"],
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
            labels=["0", "1-25", "26-99", "100-499", "500+"],
        )
        ax.set_xlabel("Reduced Dimension 1")
        ax.set_ylabel("Reduced Dimension 2")
        ax.set_zlabel("Reduced Dimension 3")
        ax.set_box_aspect(None, zoom=0.9)
        scatter.set_sizes([0.1])

    plt.savefig(get_file_path(f"Data/{title}.png"))
    plt.show()


allow_visualize = input(
    "Type visualize to re-compute node embedding visualizations, or press enter to show pre-computed visualizations: "
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
else:
    for image in [
        "Broker Score (2D PCA)",
        "Broker Score (3D PCA)",
        "Broker Score (2D t-SNE)",
        "Broker Score (3D t-SNE)",
    ]:
        plt.imshow(mpimg.imread(get_file_path(f"Data/{image}.png")))
        plt.show()


# %%
# Define model training/evaluation functions
def train_eval(model, prediction_type="both", **kwargs):
    set_random_seed(0)
    if prediction_type == "classification":
        model.fit(X_train, Y_train_class, **kwargs)
    else:
        model.fit(X_train, Y_train, **kwargs)
    scores = eval(model, prediction_type)
    return model, scores


def eval(model, prediction_type):
    scores = {"r2": None, "f1": None}
    Y_test_pred = model.predict(X_test)

    if prediction_type in ["both", "regression"]:
        r2 = r2_score(Y_test, Y_test_pred)
        scores["r2"] = r2
        print(f"    R2 score on test: {r2}")
        for i in range(10):
            print(f"        Actual: {Y_test.iloc[i]}, Predicted: {Y_test_pred[i]}")
    if prediction_type in ["both", "classification"]:
        if prediction_type == "classification":
            Y_test_pred_class = Y_test_pred
        else:
            Y_test_pred_class = pd.cut(
                Y_test_pred,
                bins=[-float("inf"), 0, 25, 100, 500, float("inf")],
                labels=[0, 1, 2, 3, 4],
            ).astype(int)
        f1 = f1_score(Y_test_class, Y_test_pred_class, average="weighted")
        scores["f1"] = f1
        print(f"    F1 score on test: {f1}")
        for i in range(10):
            print(
                f"        Actual: {Y_test_class.iloc[i]}, Predicted: {Y_test_pred_class[i]}"
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
            if prediction_type == "classification":
                layers.append(tf.keras.layers.Dense(size, activation="softmax"))
            else:
                layers.append(tf.keras.layers.Dense(size, activation="relu"))
        else:
            if prediction_type == "classification":
                layers.append(tf.keras.layers.Dense(size, activation="relu"))
            else:
                layers.append(tf.keras.layers.Dense(size))
    model = nn_model(layers)
    if prediction_type == "classification":
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# %%
# Linear regression
print("Linear Regression")
linear_regression = train_eval(linear_model.LinearRegression(), prediction_type="both")

# train_eval(linear_model.LogisticRegressionCV(cv=2, max_iter=3), X_df, Y, train_size=0.2)
#   Takes a long time to run, tries to allocate too much memory (~50 GB) if attempted on full training set

# %%
print("Random Forest Classifier")
rf_classifier, rf_classifier_score = train_eval(
    RandomForestClassifier(n_estimators=25, random_state=0),
    prediction_type="classification",
)
# %%
print("K-Nearest Neighbors Classifier")
knn_classifier, knn_classifier_score = train_eval(
    KNeighborsClassifier(n_neighbors=5), prediction_type="classification"
)
# %%
# Sequential Neural Network Regression
print("Sequential Neural Network (regression) [64, 32, 1]")
snn, snn_score = train_eval(
    create_tf_model(X_df.shape[1], layer_sizes=[64, 32, 1], prediction_type="both"),
    validation_data=(X_test, Y_test),
    epochs=100,
    batch_size=16384,
    verbose=0,
    prediction_type="both",
)
# Batch size greatly increases training speed - lower if memory issues arise

# %%
# Sequential Neural Network Classifiers
print("Sequential Neural Network (classifier) [64, 48, 5] (trial)")
snn_classify_1, snn_classify_score_1 = train_eval(
    create_tf_model(
        X_df.shape[1],
        layer_sizes=[64, 48, 48, 5],
        # Final layer size must match number of classes - largest output weight is chosen
        prediction_type="classification",
    ),
    validation_data=(X_test, Y_test_class),
    epochs=100,
    batch_size=16384,
    verbose=1,
    prediction_type="classification",
)

print("Sequential Neural Network (classifier) [144, 144, 144, 144, 144, 5]")
snn_classify_2, snn_classify_score_2 = train_eval(
    create_tf_model(
        X_df.shape[1],
        layer_sizes=[144, 144, 144, 144, 144, 5],
        prediction_type="classification",
    ),
    validation_data=(X_test, Y_test_class),
    epochs=100,
    batch_size=16384,
    verbose=1,
    prediction_type="classification",
)

# %%
# Ensemble Classifier
print(
    "Ensemble Classifier: Sequential Neural Network, Random Forest, K-Nearest Neighbors"
)
snn_classify_2, snn_classify_score_2 = train_eval(
    ensemble_model([snn_classify_2, rf_classifier, knn_classifier]),
    prediction_type="classification",
)

# And try plotting the predicted classes to compare with actual values
# Maybe save models as file for later use w/o retraining
# Somewhere plot the broker score distribution itself

# %%
