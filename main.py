# %%
# Imports
import os
import igraph as ig
import pandas as pd
import graph_tool.all as gt
from deepgl import DeepGL

# %%
# Main

# Next, work on predicting time series of broker scores from first 90% of timestamps

def get_file_path(file_name):
    path = f"{file_name}"
    if not os.path.exists(path):
        path = f"/your_code/{path}"
    return path

with open(get_file_path("Data/higgs-activity_time.txt")) as activity_time:
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
ig_graph: ig.Graph = ig.Graph.DataFrame(retweet_df, directed=True)
print(ig_graph.summary())
# %%
def example():
    # Print node 8 - user whose ID is 8 (as seen on line 6 of input file)
    print(ig_graph.vs[8])

    # Print the edges that enter node 8 - incoming retweets (user 8 made a popular post)
    incoming_edges = ig_graph.es.select(_target=ig_graph.vs[8].index)
    print(
        f"\nUser 8 has {len(incoming_edges)} incoming retweets (source spreader score)"
    )

    # Print the edges that leave node 8 - outgoing retweets (user 8 did not retweet)
    outgoing_edges = ig_graph.es.select(_source=ig_graph.vs[8].index)
    print(f"\nUser 8 has {len(outgoing_edges)} outgoing retweets")


# %%
def calculate_source_spreader_score(graph: ig.Graph, index: int) -> int:
    return graph.degree(index, mode="in")


def calculate_broker_score(graph: ig.Graph, index) -> int:
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

def igraph_to_graphtool(graph: ig.Graph) -> gt.Graph:
    # Converts an igraph graph to a graphtool graph
    #   Much faster using intermediate file
    intermediate_file = get_file_path("Data/igraph_higgs.graphml")
    graph.write_graphml(intermediate_file)
    gt_graph = gt.load_graph(intermediate_file)
    if os.path.exists(intermediate_file):
        os.remove(intermediate_file)
    return gt_graph

recompute = input(
    "Type recompute to recompute source spreader and broker scores, or press enter to use precomputed scores: "
)
if recompute.lower() == "recompute":
    user_df = pd.DataFrame({"index": [node.index for node in ig_graph.vs]})
    user_df["source_spreader_score"] = user_df["index"].apply(
        calculate_source_spreader_score
    )
    user_df["broker_score"] = user_df["index"].apply(calculate_broker_score)
    user_df.to_csv("Data/precomputed_scores.csv", index=False)
else:
    user_df = pd.read_csv(get_file_path("Data/precomputed_scores.csv"))
print(user_df.head())
print(user_df.shape)

# Add source_spreader_score and broker_score as node attributes of the graph
ig_graph.vs["source_spreader_score"] = user_df["source_spreader_score"]
ig_graph.vs["broker_score"] = user_df["broker_score"]
print(ig_graph.summary())

# Convert igraph graph to graphtool graph
gt_graph = igraph_to_graphtool(ig_graph)
print(gt_graph)

# %%
def deepgl_example(graph: gt.Graph):
    # DeepGL setting
    deepgl = DeepGL(
        base_feat_defs=[
            'in_degree',
            'out_degree',
            # 'total_degree',
            'pagerank',
            # 'betweenness',
            # 'closeness',
            # 'eigenvector',
            # 'katz',
            'hits',
            # 'kcore_decomposition',
            # 'sequential_vertex_coloring',
            # 'max_independent_vertex_set',
            # 'label_components',
            # 'label_out_component',
            # 'label_largest_component',
            'source_spreader_score', # Custom node attribute
            'broker_score' # Custom node attribute
        ],
        ego_dist=1, # Runs much faster with lower values - 1 runs quickly, default of 3
        nbr_types=['all'],
        lambda_value=0.9,
        transform_method='log_binning'
    )
    X1 = deepgl.fit_transform(graph)
    print("This is a node embedding matrix of each node in the graph")
    print(X1)
    print(X1.shape)
    # Ideally run this at least once with good parameters and then cache results
    # These feature vectors can easily be used for any ML purposes
deepgl_example(gt_graph)
# %%
