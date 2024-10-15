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

df = pd.DataFrame(data, columns=['retweeter', 'poster', 'timestamp', 'activity_type'])
df = df.sort_values(by='timestamp') # Sort by timestamp
df = df[df['activity_type'] == 'RT'] # Keep only retweets
df = df.drop(columns=['activity_type']) # Drop activity_type column
print(df)


# %%
graph: ig.Graph = ig.Graph.DataFrame(df, directed=True)
print(graph.summary())
# %%
# Print node 8 - user whose ID is 8 (as seen on line 6 of input file)
print(graph.vs[8])

# Print the edges that enter node 8 - incoming retweets (user 8 made a popular post)
incoming_edges = graph.es.select(_target=graph.vs[8].index)
print(f"\nUser 8 has {len(incoming_edges)} incoming retweets (source spreader score)")
print([e for e in incoming_edges])

# Print the edges that leave node 8 - outgoing retweets (user 8 did not retweet)
outgoing_edges = graph.es.select(_source=graph.vs[8].index)
print(f"\nUser 8 has {len(outgoing_edges)} outgoing retweets")
print([e for e in outgoing_edges])
# %%
