
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# hold 

# sizes = [75, 75, 300]
# probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]

sizes = [50, 50, 50]
probs = [[0.1, 0.001, 0.001], [0.001, 0.1, 0.001], [0.001, 0.001, 0.1]]

random_seed = 42
np.random.seed(random_seed)
g = nx.stochastic_block_model(sizes, probs)


# guarantee completedness
while not nx.is_connected(g):
    component1, component2 = random.sample(list(nx.connected_components(g)), 2)
    node1 = random.choice(list(component1))
    node2 = random.choice(list(component2))
    
    g.add_edge(node1, node2)

# H = nx.quotient_graph(g, g.graph["partition"], relabel=True)
fh = open("./graphs/size_fifty/10_point1.txt", "wb")
nx.write_edgelist(g, fh, data=False)
nx.draw(g, with_labels = True)
# nx.draw(H, with_labels = True)
plt.savefig("./graphs/size_fifty_pngs/10_point1.png")
plt.show()
