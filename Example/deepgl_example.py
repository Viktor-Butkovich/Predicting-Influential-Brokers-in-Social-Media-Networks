import graph_tool.all as gt
from deepgl import DeepGL

# Data from http://www.sociopatterns.org/datasets/primary-school-cumulative-networks/
g1 = gt.load_graph("./example_data/sp_data_school_day_2_g.gml")

deepgl = DeepGL(
    base_feat_defs=[
        "total_degree",
        "eigenvector",
        "katz",
        "pagerank",
        "closeness",
        "betweenness",  # , 'gender'
    ],
    ego_dist=1,
    nbr_types=["all"],
    lambda_value=0.7,
    transform_method="log_binning",
)

X1 = deepgl.fit_transform(g1)

print(X1)
print(X1.shape)
for nth_layer_feat_def in deepgl.feat_defs:
    print(nth_layer_feat_def)

g2 = gt.load_graph("./example_data/sp_data_school_day_1_g.gml")
X2 = deepgl.transform(g2)

print(X2)
print(X2.shape)
