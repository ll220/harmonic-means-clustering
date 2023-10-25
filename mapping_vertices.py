import random
random.seed(246)        # or any integer
import numpy as np
# np.random.seed(4812)
import networkx as nx
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import sys
import os

# CANNOT BE CHANGED WITHOUT CHANGING CODE
NUM_CLUSTERS = 3

K_HARMONICS = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 100]

def calculate_variance(cluster_labels, cluster_centroids, position_encoding):
    cluster_inertias = []

    for i in range(len(cluster_centroids)):
        cluster_points = position_encoding[:][cluster_labels == i]  # Data points in the i-th cluster
        centroid = cluster_centroids[i]  # Centroid of the i-th cluster

        distances_squared = np.square(cluster_points - centroid)
        cluster_inertia = np.sum(distances_squared)  # Calculate inertia for the i-th cluster
        cluster_inertia = np.sqrt(cluster_inertia)
        cluster_inertias.append(cluster_inertia)

    print(cluster_inertias)
    return cluster_inertias

# def get_k_harmonic_distances(laplacian_inverse, nodes, n):
#     print(laplacian_inverse)
#     distances = np.zeros((n, n))

#     for u in range(n):
#         for v in range(u):
#             vector = np.zeros((n, 1))
#             vector[u][0] = 1
#             vector[v][0] = -1

#             intermediate = np.copy(laplacian_inverse)

#             for i in range(K-1):
#                 intermediate = np.matmul(intermediate, laplacian_inverse)

#             distance = np.matmul(vector.transpose(), intermediate)
#             distance = (np.matmul(distance, vector))[0][0]
#             distance = np.sqrt(distance)
#             print("u:", u+1, " v:", v+1, " distance:", distance)

#             distances[v][u] = distance
#     return distances

# def get_harmonic_distances(position_matrix, n):
#     print(position_matrix)
#     distances = np.zeros((n, n))

#     for u in range(n):
#         u_position = position_matrix[:,u]
#         for v in range(u):
#             v_position = position_matrix[:,v]

#             print(u_position)
#             print(v_position)

#             distance = np.linalg.norm(u_position - v_position)
#             print("u:", u+1, " v:", v+1, " distance:", distance)
#             distances[v][u] = distance
#     return distances

def get_expected_clustering(nodes):
    expected_labels = []
    for cluster in range(NUM_CLUSTERS):
        cluster = [cluster for x in range(0, int(len(nodes) / 3))]
        expected_labels.extend(cluster)

    return expected_labels

def return_clustering(k, G, num_clusters, output_png=None):

    laplacian = nx.laplacian_matrix(G)
    laplacian = laplacian.toarray()

    laplacian_inverse = np.linalg.pinv(laplacian) 

    position_encoding = scipy.linalg.fractional_matrix_power(laplacian_inverse, float(k / 2))
    position_encoding = position_encoding.real
    position_encoding = position_encoding.transpose()


    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(position_encoding)
    # print(kmeans.labels_)

    colors = ['red', 'blue', 'green']
    node_colors = [colors[cluster] for cluster in kmeans.labels_]

    nx.draw(G, with_labels=True, node_color=node_colors)
    plt.title('Clustered Graph')
    # plt.savefig(output_png)
    # plt.show()
    plt.clf()


    # cluster_inertias = calculate_inertias(kmeans, position_encoding)
    # print(kmeans.cluster_centers_)
    # print(cluster_inertias)

    # f = open(output_file, "w")
    # f.write(np.array2string(kmeans.cluster_centers_))
    # f.write(str(cluster_inertias))
    # f.close()

    return kmeans, position_encoding

def get_triangle_lengths(cluster_centers):
    lengths = []

    lengths.append((np.linalg.norm(cluster_centers[1] - cluster_centers[0]), (0, 1)))
    lengths.append((np.linalg.norm(cluster_centers[2] - cluster_centers[1]), (1, 2)))
    lengths.append((np.linalg.norm(cluster_centers[2] - cluster_centers[0]), (0, 2)))
    lengths.sort()

    for i in range(3):
        not_normalized = lengths[i]
        lengths[i] = (not_normalized[0] / lengths[2][0], not_normalized[1])

    return lengths


def plot_triangle(sorted_triangle_lengths, output_file_name, variances):

    # use law of cosines
    B = np.arccos((sorted_triangle_lengths[0][0]**2 + sorted_triangle_lengths[2][0]**2 - sorted_triangle_lengths[1][0]**2) / (2 * sorted_triangle_lengths[0][0] * sorted_triangle_lengths[2][0]))
    
    P0 = (0, 0)
    P1 = (sorted_triangle_lengths[0][0], 0)
    P2 = (sorted_triangle_lengths[2][0] * np.cos(B), sorted_triangle_lengths[2][0] * np.sin(B))

    x_vertices = [P0[0], P1[0], P2[0], P0[0]]
    y_vertices = [P0[1], P1[1], P2[1], P0[1]]

    ax = plt.gca()


    # ax.set_xlim = ((0, 1))
    # ax.set_ylim = ((0, 1))
    # ax.set_aspect('equal', adjustable='box')  # Equal aspect ratio

    ax.plot(x_vertices, y_vertices)

    first_vertex_index = (set(sorted_triangle_lengths[0][1])).intersection(set(sorted_triangle_lengths[2][1])).pop()
    second_vertex_index = (set(sorted_triangle_lengths[0][1])).intersection(set(sorted_triangle_lengths[1][1])).pop()
    third_vertex_index = (set(sorted_triangle_lengths[1][1])).intersection(set(sorted_triangle_lengths[2][1])).pop()

    variance_1 = plt.Circle(P0, variances[first_vertex_index], fill=False)
    variance_2 = plt.Circle(P1, variances[second_vertex_index], fill=False)
    variance_3 = plt.Circle(P2, variances[third_vertex_index], fill=False)

    ax.add_patch(variance_1)
    ax.add_patch(variance_2)
    ax.add_patch(variance_3)

    ax.set_title("Centers Triangle")
    # plt.savefig(output_file_name)
    plt.show()
    plt.clf()

def plot_k_vs_accuracy(k_harmonics, accuracies, output_file_name):
    plt.plot(k_harmonics, accuracies)
    plt.title("k vs adjusted random score")
    plt.savefig(output_file_name)
    # plt.show()
    plt.clf()



def analysis(input_file, output_directory):
    print("Now analyzing: ", input_file)
    # assumption that all vertices are encoded as 0-n consecutive integers
    f = open(input_file, "r")
    file_string = f.read()
    f.close()
    edges = file_string.split('\n')

    unordered_g = nx.parse_edgelist(edges, nodetype=int)

    G = nx.Graph()
    G.add_nodes_from(sorted(unordered_g.nodes(data=True)))
    G.add_edges_from(unordered_g.edges(data=True))


    nodes = list(G.nodes)

    if not nx.is_connected(G):
        print("geh")
        quit()

    true_clusters = get_expected_clustering(nodes)
    accuracies = []

    for k in K_HARMONICS:
        print("k: ", k)
        accuracy_trials = []
        triangle_trials = []
        max_accuracy = -2
        labels = centers = best_position_encoding = None

        if k==0.1: 
            file_name = "point_1"
        elif k==0.5:
            file_name = "point_5"
        else:
            file_name = str(k)

        for _ in range(10):
            kmeans, position_encoding = return_clustering(k, G, NUM_CLUSTERS)
            accuracy = adjusted_rand_score(true_clusters, kmeans.labels_)
            triangle = get_triangle_lengths(kmeans.cluster_centers_)

            accuracy_trials.append(accuracy)
            triangle_trials.append(triangle)

            if (accuracy > max_accuracy):
                max_accuracy =  accuracy 
                labels = kmeans.labels_
                centers = kmeans.cluster_centers_
                best_position_encoding = position_encoding

        mean_accuracy = np.mean(accuracy_trials)
        print("Mean accuracy: ", mean_accuracy)
        accuracies.append(mean_accuracy)


        max_accuracy_index = accuracy_trials.index(max(accuracy_trials))
        variances = calculate_variance(labels, centers, best_position_encoding)

        if (accuracy_trials[max_accuracy_index] == 1):
            triangle_file_name = (os.path.basename(input_file))[:-4] + "_" + file_name + "_triangle_(total_accuracy).png"
        else:
            triangle_file_name = (os.path.basename(input_file))[:-4] + "_" + file_name + "_triangle.png"

        best_triangle = triangle_trials[max_accuracy_index]

        plot_triangle(best_triangle, os.path.join(output_directory, triangle_file_name), variances)

    k_vs_accuracy_filename = (os.path.basename(input_file))[:-4] + "_k_vs_accuracy.png"
    plot_k_vs_accuracy(K_HARMONICS, accuracies, os.path.join(output_directory, k_vs_accuracy_filename))

def main():   
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    input_files = os.listdir(input_directory)

    for input_file in input_files:
        analysis(os.path.join(input_directory, input_file), output_directory)

if __name__ == "__main__":
    main()
