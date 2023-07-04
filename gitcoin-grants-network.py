import os

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import plotly.graph_objs as go
import plotly.express as px
import locale
import networkx as nx
import time
import json

PATH_TO_DATA = 'data'

st.set_page_config(
    page_title="Gitcoin Beta Rounds",
    page_icon="📊",
    layout="wide",

)

st.title('Gitcoin Rounds Cluster Analysis')
st.write('This chart aims to help verify cluster from a clustering method.')
st.write('')


def get_first_file_with_pattern(path, pattern):
    files = os.listdir(path)
    files = [f for f in files if pattern in f]
    return files[0]


@st.cache_data(ttl=3600)
def load_transaction_data():
    file = get_first_file_with_pattern(PATH_TO_DATA, 'tx_')
    df = pd.read_parquet(os.path.join(PATH_TO_DATA, file))
    return df


@st.cache_data(ttl=3600)
def load_votes_data():
    file = get_first_file_with_pattern(PATH_TO_DATA, 'votes_')
    df = pd.read_csv(os.path.join(PATH_TO_DATA, file))
    return df


@st.cache_data(ttl=3600)
def load_projects_data():
    file = get_first_file_with_pattern(PATH_TO_DATA, 'projects_')
    df = pd.read_csv(os.path.join(PATH_TO_DATA, file))
    return df


@st.cache_data(ttl=3600)
def load_features_voters_data():
    file = get_first_file_with_pattern(PATH_TO_DATA, 'voters_features_')
    df = pd.read_csv(os.path.join(PATH_TO_DATA, file))
    return df


data_load_state = st.text('Loading data...')
df_tx = load_transaction_data()
df_v = load_votes_data()
# df_p = load_projects_data()
df_fv = load_features_voters_data()
data_load_state.text("")


list_clusters_address = df_fv.loc[df_fv['has_lcs'], 'address'].unique()

address = st.selectbox('Select address', list_clusters_address)

cluster = df_fv.loc[df_fv['address'] == address, 'lcs'].values[0]
df_cluster = pd.DataFrame.from_dict(json.loads(cluster))

cluster_address = df_cluster['address'].unique()

# load transactions of that cluster
df_tx_cluster = df_tx.loc[df_tx['EOA'].isin(cluster_address)]
df_v_cluster = df_v.loc[df_v['voter'].isin(cluster_address)]

# Build the graph from the transactions
G = nx.from_pandas_edgelist(df_tx_cluster, 'from_address', 'to_address', True, create_using=nx.DiGraph())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(label="Addresses", value=G.number_of_nodes())
col2.metric(label="Voters", value=df_v_cluster['voter'].nunique())
col3.metric(label="Votes", value=df_v_cluster['voter'].count())
col4.metric(label="Projects", value=df_v_cluster['project'].nunique())
col5.metric(label="Transactions", value=df_tx_cluster.count()[0])


# Create a dictionary to store the node labels
labels = {}
colors = []
# Set the node labels to display only the first 4 characters
for node in G.nodes():
    labels[node] = node[:4]
    if node in df_v_cluster['voter'].unique():
        colors.append('red')
    elif node == "0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc":
        colors.append('green')
    else:
        colors.append('blue')

line_color = '#008F11'

# Compute the layout
current_time = time.time()
pos = nx.spring_layout(G, dim=3, k=.09, iterations=50)
new_time = time.time()

# Extract node information
node_x = [coord[0] for coord in pos.values()]
node_y = [coord[1] for coord in pos.values()]
node_z = [coord[2] for coord in pos.values()]  # added z-coordinates for 3D
node_names = list(pos.keys())

# Compute the degrees of the nodes 
degrees = np.array([G.degree(node_name) for node_name in node_names])
# Apply the natural logarithm to the degrees 
log_degrees = np.log(degrees + 1)
node_sizes = log_degrees * 10

# Extract edge information
edge_x = []
edge_y = []
edge_z = []
# edge_weights = []

for edge in G.edges(data=True):
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])
    # edge_weights.append(edge[2]['amountUSD'])

# Create the edge traces
edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    line=dict(width=1, color=line_color),
    hoverinfo='none',
    mode='lines',
    marker=dict(opacity=0.5))

# Create the node traces
node_trace = go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        color=colors,
        size=node_sizes,
        opacity=1,
        sizemode='diameter'
    ))

node_adjacencies = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))

# Prepare text information for hovering
node_trace.text = [f'{name}: {adj} connections' for name, adj in zip(node_names, node_adjacencies)]

# Create the figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='3D Network graph of cluster of voters',
                    titlefont=dict(size=20),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        showarrow=False,
                        text="This graph shows the connections between a cluster of voters and grants based on "
                             "donations and transactions.",
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002)],
                    scene=dict(
                        xaxis_title='X Axis',
                        yaxis_title='Y Axis',
                        zaxis_title='Z Axis')))

st.plotly_chart(fig, use_container_width=True)
st.caption('Time to compute layout: ' + str(round(new_time - current_time, 2)) + ' seconds')
