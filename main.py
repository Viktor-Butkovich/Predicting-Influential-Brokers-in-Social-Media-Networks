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

df = pd.DataFrame(data, columns=["retweeter", "poster", "timestamp", "activity_type"])
df = df.sort_values(by="timestamp")  # Sort by timestamp
df = df[df["activity_type"] == "RT"]  # Keep only retweets
df = df.drop(columns=["activity_type"])  # Drop activity_type column


# %%
graph: ig.Graph = ig.Graph.DataFrame(df, directed=True)
print(graph.summary())
# %%
# Print node 8 - user whose ID is 8 (as seen on line 6 of input file)
print(graph.vs[8])

# Print the edges that enter node 8 - incoming retweets (user 8 made a popular post)
incoming_edges = graph.es.select(_target=graph.vs[8].index)
print(f"\nUser 8 has {len(incoming_edges)} incoming retweets (source spreader score)")

# Print the edges that leave node 8 - outgoing retweets (user 8 did not retweet)
outgoing_edges = graph.es.select(_source=graph.vs[8].index)
print(f"\nUser 8 has {len(outgoing_edges)} outgoing retweets")

# %%
def calculate_source_spreader_score(index):
    return len(graph.es.select(_target=graph.vs[8].index))


def calculate_broker_score(index, verbose=False):
    outgoing_edges = graph.es.select(_source=graph.vs[index].index)
    broker_score = 0
    if verbose:
        print(f"\nUser {index} has {len(outgoing_edges)} outgoing retweets")
    for edge in outgoing_edges:
        source = graph.vs[edge.source].index
        target = graph.vs[edge.target].index
        if verbose:
            print(f"Edge from {source} to {target}")
        target_incoming_edges = graph.es.select(_target=graph.vs[target].index)
        timestamp = edge["timestamp"]
        if verbose:
            print(f"Timestamp of edge from {source} to {target}: {timestamp}")
            print(f"{target} has {len(target_incoming_edges)} incoming retweets")
        for cascading_retweet in target_incoming_edges:
            cascading_timestamp = cascading_retweet["timestamp"]
            cascading_source = graph.vs[cascading_retweet.source].index
            if (
                cascading_timestamp > timestamp and cascading_source != source
            ):  # If retweet after source's retweet and retweet isn't from source
                broker_score += 1
        if verbose:
            print()
    return broker_score


# %%
index = 120661
print(f"Node {index} source spreader score: {calculate_source_spreader_score(index)}")
print(f"Node {index} broker score: {calculate_broker_score(index)}")
# %%
