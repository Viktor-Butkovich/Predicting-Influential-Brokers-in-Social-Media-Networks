# %%
# Imports
import os
import igraph as ig
import pandas as pd
import numpy as np
import graph_tool.all as gt
import networkx as nx
from deepgl import DeepGL
import deepgl_utils
from node2vec import Node2Vec

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
        lambda index: calculate_source_spreader_score(ig_graph, index)
    )

    user_df["broker_score"] = user_df["index"].apply(
        lambda index: calculate_broker_score(ig_graph, index)
    )
    user_df.to_csv("Data/precomputed_scores.gz", index=False, compression="gzip")
else:
    user_df = pd.read_csv(
        get_file_path("Data/precomputed_scores.gz"), compression="gzip"
    )
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
def fixed_search_rel_func_space(deepgl, g, diffusion_iter=0, transform="log_binning"):
    """
    Searching the relational function space (Sec. 2.3, Rossi et al., 2018)
    """
    n_rel_feat_ops = len(deepgl.rel_feat_ops)
    n_nbr_types = len(deepgl.nbr_types)

    for l in range(1, deepgl.ego_dist):
        prev_feat_defs = deepgl.feat_defs[l - 1]
        new_feat_defs = []

        for i, op in enumerate(deepgl.rel_feat_ops):
            for j, nbr_type in enumerate(deepgl.nbr_types):
                for k, prev_feat_def in enumerate(prev_feat_defs):
                    new_feat_def = deepgl._comp_rel_op_feat(
                        g, op, nbr_type, prev_feat_def
                    )

                    new_feat = np.expand_dims(
                        g.vertex_properties[new_feat_def].a, axis=1
                    )
                    deepgl.X = np.concatenate((deepgl.X, new_feat), axis=1)
                    new_feat_defs.append(new_feat_def)

        deepgl.feat_defs.append(new_feat_defs)

        deepgl_utils.Processing.feat_diffusion(deepgl.X, g, iter=diffusion_iter)

        if transform == "log_binning":
            deepgl_utils.Processing().log_binning(
                deepgl.X, alpha=deepgl.log_binning_alpha
            )

        # Added nan processing
        deepgl.X = np.nan_to_num(deepgl.X, nan=0.0, posinf=0.0, neginf=0.0)

        # feature pruning
        deepgl.X, deepgl.feat_defs = deepgl_utils.Processing().prune_feats(
            deepgl.X, deepgl.feat_defs, lambda_value=deepgl.lambda_value
        )

    return deepgl


def get_deepgl_embeddings(graph: gt.Graph):
    # DeepGL setting
    deepgl = DeepGL(
        base_feat_defs=[
            "in_degree",
            "out_degree",
            # 'total_degree',
            "pagerank",
            "betweenness",
            "closeness",
            "eigenvector",
            "katz",
            # 'hits', # HITS doesn't seem to be fully implemented in deepgl
            # 'kcore_decomposition',
            # 'sequential_vertex_coloring',
            # 'max_independent_vertex_set',
            # 'label_components',
            # 'label_out_component',
            # 'label_largest_component',
            # 'source_spreader_score', # Custom node attribute
            # 'broker_score' # Custom node attribute
        ],
        ego_dist=1,  # Runs much faster with lower values - 1 runs quickly, default of 3
        nbr_types=["all"],
        lambda_value=0.9,
        transform_method="log_binning",
    )

    # The following is equivalent to X = deepgl.fit_transform(graph) but with nan handling
    deepgl._prepare_base_feats(graph, transform=deepgl.transform_method)

    fixed_search_rel_func_space(
        deepgl,
        graph,
        diffusion_iter=deepgl.diffusion_iter,
        transform=deepgl.transform_method,
    )
    return deepgl.X


def get_node2vec_embeddings(
    G: gt.Graph, d: int, r: int, k: int, l: int, p: int, q: int
) -> pd.DataFrame:
    """
    Description:
        Generates node embeddings using the Node2Vec algorithm
    Input:
        G: Graph to generate embeddings for
        X: Node attributes to append to the embeddings
        d: Dimensionality of the embeddings, default of 128
        r: Number of walks per node, default of 10
        k: Neighborhood (window) size, default of 10
        l: Walk length, default of 80
        p: In-out parameter, default of 1
        q: Return parameter, default of 1
    Output:
        DataFrame containing node embeddings
    """
    # Initialize and fit Node2Vec with desired parameters
    # Convert graph-tool graph to networkx graph
    G_nx = nx.Graph()
    for v in G.vertices():
        G_nx.add_node(int(v))
    for e in G.edges():
        G_nx.add_edge(int(e.source()), int(e.target()))

    # Initialize and fit Node2Vec with desired parameters
    node2vec = Node2Vec(
        G_nx, dimensions=d, num_walks=r, walk_length=l, p=p, q=q, seed=0, quiet=False
    )
    model = node2vec.fit(window=k, min_count=1, batch_words=4)

    # Transform nodes to embeddings - note that node2vec does not include node attributes in the embeddings, so these would need to be added afterward
    embeddings = {str(node): model.wv[str(node)] for node in G_nx.nodes()}

    return pd.DataFrame(embeddings).T


# %%
recompute = input(
    "Type recompute to recompute node embeddings, or press enter to use precomputed node embeddings: "
)
if recompute.lower() == "recompute":
    use_deepgl = input(
        "Type deepgl to compute deepgl emebeddings, or press enter to use node2vec embeddings: "
    )
    if use_deepgl.lower() == "deepgl":
        X = get_deepgl_embeddings(gt_graph)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    else:
        limit_nodes = False
        if limit_nodes:
            # Randomly remove all but 10,000 nodes from gt_graph for faster testing
            nodes_to_keep = np.random.choice(
                gt_graph.get_vertices(), 10000, replace=False
            )
            nodes_to_remove = set(gt_graph.get_vertices()) - set(nodes_to_keep)
            gt_graph.remove_vertex(list(nodes_to_remove), fast=True)
        X_df = get_node2vec_embeddings(gt_graph, 128, 4, 5, 10, 1, 1)  # d=16
    X_df.to_csv(
        get_file_path("Data/precomputed_node_embeddings_128.gz"),
        index=False,
        compression="gzip",
    )
else:
    X_df = pd.read_csv(
        get_file_path("Data/precomputed_node_embeddings_128.gz"), compression="gzip"
    )
print("This is a node embedding matrix of each node in the graph")
print(X_df.head())
# %%
