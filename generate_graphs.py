import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.neighbors import kneighbors_graph

def load_iris_graph_and_labels(num_neighbors : int) -> nx.Graph:
    data, labels = load_iris(return_X_y=True)
    points = np.array(data)
    adjacency_matrix = kneighbors_graph(points, num_neighbors, mode='connectivity', include_self=False)
    graph = nx.from_numpy_array(adjacency_matrix)
    return graph, labels

def load_wine_graph_and_labels(num_neighbors : int) -> nx.Graph:
    data, labels = load_wine(return_X_y=True)
    points = np.array(data)
    adjacency_matrix = kneighbors_graph(points, num_neighbors, mode='connectivity', include_self=False)
    graph = nx.from_numpy_array(adjacency_matrix)
    return graph, labels

def load_cancer_graph_and_labels(num_neighbors : int) -> nx.Graph:
    data, labels = load_breast_cancer(return_X_y=True)
    points = np.array(data)
    adjacency_matrix = kneighbors_graph(points, num_neighbors, mode='connectivity', include_self=False)
    graph = nx.from_numpy_array(adjacency_matrix)
    return graph, labels

def load_uci_graph_and_labels(
    num_neighbors : int, 
    id : int
):
    from ucimlrepo import fetch_ucirepo 
    dataset = fetch_ucirepo(id=id)
    X = dataset.data.features.to_numpy()
    y = dataset.data.targets.to_numpy()[:,0]
    adjacency_matrix = kneighbors_graph(X, num_neighbors, mode='connectivity', include_self=False)
    graph = nx.from_numpy_array(adjacency_matrix)
    return graph, y

def load_block_stochastic_graph_and_labels(
    num_nodes_per_cluster : int,
    num_clusters : int,
    intercluster_p : float,
    intracluster_p : float
) -> nx.Graph:
    """ return a block stochatsic graph
    
    args:
        num_nodes_per_cluster
        num_clusters
        intercluster_p  : probability of edge between nodes in different clusters
        intracluster_p : probability of edge between nodes in same cluster

    """
    cluster_sizes = num_clusters * [num_nodes_per_cluster]
    P = (intracluster_p-intercluster_p)*np.eye(num_clusters) + (intercluster_p)*np.ones((num_clusters, num_clusters))
    graph = nx.stochastic_block_model(
        cluster_sizes,
        P
    )
    labels = np.array([
        i 
        for i in range(num_clusters)
        for _ in range(num_nodes_per_cluster)
    ])

    return graph, labels


if __name__=="__main__":
    """
    graph, labels = load_cancer_graph_and_labels(num_neighbors=25)
    colors = ['red', 'blue', 'green']
    node_colors = [colors[cluster] for cluster in labels]
    nx.draw(graph, with_labels=True, node_color=node_colors)
    plt.title('Iris Generated Graph k=125')
    # plt.savefig('Iris Generated Graph k=125')
    plt.show()
    """
    graph, labels = load_uci_graph_and_labels(num_neighbors=10, id=320)

# sizes = [75, 75, 300]
# # probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]

# sizes = [50] * 10
# # probs = [[0.1, 0.001, 0.001], [0.001, 0.1, 0.001], [0.001, 0.001, 0.1]]

# random_seed = 42
# seed = np.random.seed(random_seed)
# g = nx.random_partition_graph(sizes, 1, 0.02, seed=seed)
# # g = nx.stochastic_block_model(sizes, probs)


# # guarantee completedness
# while not nx.is_connected(g):
#     component1, component2 = random.sample(list(nx.connected_components(g)), 2)
#     node1 = random.choice(list(component1))
#     node2 = random.choice(list(component2))
    
#     g.add_edge(node1, node2)

# H = nx.quotient_graph(g, g.graph["partition"], relabel=True)
# fh = open("./ten_graphs/size_50/100_2.txt", "wb")
# nx.write_edgelist(g, fh, data=False)
# nx.draw(g, with_labels = True)
# nx.draw(H, with_labels = True)
# # plt.savefig("./graphs/size_fifty_pngs/10_point5.png")
# plt.show()
