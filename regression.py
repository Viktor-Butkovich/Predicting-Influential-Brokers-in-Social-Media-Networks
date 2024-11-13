# %%
# Imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# %%
def get_file_path(file_name):
    path = f"{file_name}"
    if not os.path.exists(path):
        path = f"/your_code/{path}"
    return path


X_df = pd.read_csv(
    get_file_path("Data/precomputed_node_embeddings.gz"), compression="gzip"
)
user_df = pd.read_csv(get_file_path("Data/precomputed_scores.gz"), compression="gzip")

print("This is a node embedding matrix of each node in the graph")
print(X_df.head())

# %%
Y = pd.Series(user_df["broker_score"])
X_train, X_test, Y_train, Y_test = train_test_split(
    X_df, Y, train_size=0.2, shuffle=True, random_state=0
)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# %%


def train_eval(model, X, Y, train_size=0.9):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=train_size, shuffle=True, random_state=0
    )
    model.fit(X_train, Y_train)
    eval(model, X_train, Y_train, X_test, Y_test)
    return model


def eval(model, X_train, Y_train, X_test, Y_test):
    print(model)
    print(f"    Score on train: {model.score(X_train, Y_train)}")
    print(f"    Score on test: {model.score(X_test, Y_test)}")
    print("    Sample predictions: ")
    Y_pred = model.predict(X_test)
    for i in range(25):
        print(f"        Actual: {Y_test.iloc[i]}, Predicted: {Y_pred[i]}")


train_eval(linear_model.LinearRegression(), X_df, Y, train_size=0.2)

# train_eval(linear_model.LogisticRegressionCV(cv=2, max_iter=3), X_df, Y, train_size=0.2)
#   Takes a long time to run, uses too much memory if attempted on full training set

# Possibly attempt classification, between users who will have little/no broker score, meoderate broker score, and high broker score
#   Label "information brokers" with high broker score, find F1 score of this prediction


# %%
# Try running PCA and T-SNA on node embeddings to visualize node embedding graph
#   Especially with categorized Y values to try to find patterns in the graph
def visualize(X, Y, visualizer, title: str, data="all", dimensions=2):
    X_transformed = visualizer.fit_transform(X)
    if dimensions == 2:
        plt.title(title)
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=Y, s=3)
        # plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=pd.Series(y).apply(kingdom_to_color), s=3)
        plt.xlabel("Reduced Dimension 1")
        plt.ylabel("Reduced Dimension 2")

    elif dimensions == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d", title=title)
        ax.scatter(
            X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=Y, s=3
        )
        ax.set_xlabel("Reduced Dimension 1")
        ax.set_ylabel("Reduced Dimension 2")
        ax.set_zlabel("Reduced Dimension 3")
        ax.set_box_aspect(None, zoom=0.9)

    # Possible code for color labels once we add Y categories
    # handles = [mpatches.Patch(color=kingdom_to_color(kingdom), label=kingdom) for kingdom in kingdoms]
    # plt.legend(handles, kingdom_names, ncol=1, bbox_to_anchor=(1, 1))

    plt.show()


visualize(X_df, Y, PCA(n_components=2), "PCA", dimensions=2)
visualize(X_df, Y, PCA(n_components=3), "PCA", dimensions=3)

# tSNE takes longer to run, try later
# visualize(X_df, Y, TSNE(n_components=2), 't-SNE')
# %%
