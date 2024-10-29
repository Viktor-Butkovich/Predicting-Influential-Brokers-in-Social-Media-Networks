# %%
# Imports
import igraph as ig
import pandas as pd

# %%
# Main
with open("Data/higgs-activity_time.txt") as activity_time:
    data = []
    for line in activity_time:
        user1, user2, timestamp, activity_type = line.strip().split()
        data.append([int(user1), int(user2), int(timestamp), activity_type])

retweet_df = pd.DataFrame(
    data, columns=["retweeter", "poster", "timestamp", "activity_type"]
)
retweet_df = retweet_df.sort_values(by="timestamp")  # Sort by timestamp
retweet_df = retweet_df[retweet_df["activity_type"] == "RT"]  # Keep only retweets
retweet_df = retweet_df.drop(columns=["activity_type"])  # Drop activity_type column


# %%
graph: ig.Graph = ig.Graph.DataFrame(retweet_df, directed=True)
print(graph.summary())
# %%
def example():
    # Print node 8 - user whose ID is 8 (as seen on line 6 of input file)
    print(graph.vs[8])

    # Print the edges that enter node 8 - incoming retweets (user 8 made a popular post)
    incoming_edges = graph.es.select(_target=graph.vs[8].index)
    print(
        f"\nUser 8 has {len(incoming_edges)} incoming retweets (source spreader score)"
    )

    # Print the edges that leave node 8 - outgoing retweets (user 8 did not retweet)
    outgoing_edges = graph.es.select(_source=graph.vs[8].index)
    print(f"\nUser 8 has {len(outgoing_edges)} outgoing retweets")


# %%
def calculate_source_spreader_score(index):
    return graph.degree(index, mode="in")


def calculate_broker_score(index):
    outgoing_edges = graph.es.select(_source=index)
    broker_score = 0
    for edge in outgoing_edges:
        target = edge.target
        original_timestamp = edge["timestamp"]
        target_incoming_edges = graph.es.select(_target=target)
        cascading_retweets = target_incoming_edges.select(
            lambda e: e["timestamp"] > original_timestamp and e.source != index
        )
        broker_score += len(cascading_retweets)
    return broker_score


# %%
recompute = input(
    "Type recompute to recompute source spreader and broker scores, or press enter to use precomputed scores: "
)
if recompute.lower() == "recompute":
    user_df = pd.DataFrame({"index": [node.index for node in graph.vs]})
    user_df["source_spreader_score"] = user_df["index"].apply(
        calculate_source_spreader_score
    )
    user_df["broker_score"] = user_df["index"].apply(calculate_broker_score)
    user_df.to_csv("Data/precomputed_scores.csv", index=False)
else:
    user_df = pd.read_csv("Data/precomputed_scores.csv")
print(user_df.head())
# %%
