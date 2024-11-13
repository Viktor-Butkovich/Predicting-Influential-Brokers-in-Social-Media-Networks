# %%
# Imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model

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

# %%
