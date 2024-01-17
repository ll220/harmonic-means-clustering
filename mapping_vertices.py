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
from sklearn.datasets import load_iris
from sklearn.neighbors import kneighbors_graph


# CANNOT BE CHANGED WITHOUT CHANGING CODE
NUM_CLUSTERS = 3

K_HARMONICS = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 100]

def calculate_variance(cluster_labels, cluster_centroids, position_encoding):
    mean_distances = []
    max_distances = []

    for i in range(NUM_CLUSTERS):
        cluster_points = position_encoding[:,cluster_labels == i]  # Data points in the i-th cluster
        centroid = cluster_centroids[i]  # Centroid of the i-th cluster
 
        # print("centroid: ", centroid)
        # print("mean: ", np.mean(cluster_points, axis=1))

        distance_vectors = cluster_points.T - centroid.T
        # print(cluster_points)
        # print(centroid)
        # print(distance_vectors)
        distances = np.linalg.norm(distance_vectors, axis=1)
        # print(distances)
        # print(distances.shape)

        mean_distance = np.mean(distances)
        max_distance = np.amax(distances)

        mean_distances.append(mean_distance)
        max_distances.append(max_distance)

    return mean_distances, max_distances

def get_ideal_centroids(position_encoding, cluster_labels):
    cluster_centroids = np.zeros((NUM_CLUSTERS, position_encoding.shape[0]))

    for i in range(NUM_CLUSTERS):
        cluster_points = position_encoding[:,cluster_labels == i]  # Data points in the i-th cluster

        cluster_centroids[i] = np.mean(cluster_points, axis=1)

    return cluster_centroids

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
        cluster = [cluster for x in range(0, int(len(nodes) / NUM_CLUSTERS))]
        expected_labels.extend(cluster)

    return np.array(expected_labels)

def get_position_encoding(k, G):
    laplacian = nx.laplacian_matrix(G)
    laplacian = laplacian.toarray()

    laplacian_inverse = np.linalg.pinv(laplacian)
    position_encoding = scipy.linalg.fractional_matrix_power(laplacian_inverse, float(k / 2))
    position_encoding = position_encoding.real
    position_encoding = position_encoding.transpose()
    return position_encoding

def return_clustering(G, position_encoding, num_clusters, output_png=None):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(position_encoding)

    return kmeans

def get_triangle_lengths(cluster_centers):
    lengths = []

    lengths.append((np.linalg.norm(cluster_centers[1] - cluster_centers[0]), (0, 1)))
    lengths.append((np.linalg.norm(cluster_centers[2] - cluster_centers[1]), (1, 2)))
    lengths.append((np.linalg.norm(cluster_centers[2] - cluster_centers[0]), (0, 2)))
    lengths.sort()

    normalized_lengths = []
    for i in range(3):
        not_normalized = lengths[i]
        normalized_lengths.append((not_normalized[0] / lengths[2][0], not_normalized[1]))

    return lengths, normalized_lengths


def plot_triangle(sorted_triangle_lengths, output_file_name, variances=None):

    # use law of cosines
    B = np.arccos((sorted_triangle_lengths[0][0]**2 + sorted_triangle_lengths[2][0]**2 - sorted_triangle_lengths[1][0]**2) / (2 * sorted_triangle_lengths[0][0] * sorted_triangle_lengths[2][0]))
    
    P0 = (0, 0)
    P1 = (sorted_triangle_lengths[0][0], 0)
    P2 = (sorted_triangle_lengths[2][0] * np.cos(B), sorted_triangle_lengths[2][0] * np.sin(B))

    x_vertices = [P0[0], P1[0], P2[0], P0[0]]
    y_vertices = [P0[1], P1[1], P2[1], P0[1]]

    ax = plt.gca()

    ax.plot(x_vertices, y_vertices)

    if variances is not None:
        first_vertex_index = (set(sorted_triangle_lengths[0][1])).intersection(set(sorted_triangle_lengths[2][1])).pop()
        second_vertex_index = (set(sorted_triangle_lengths[0][1])).intersection(set(sorted_triangle_lengths[1][1])).pop()
        third_vertex_index = (set(sorted_triangle_lengths[1][1])).intersection(set(sorted_triangle_lengths[2][1])).pop()

        variance_1 = plt.Circle(P0, variances[first_vertex_index], fill=False)
        variance_2 = plt.Circle(P1, variances[second_vertex_index], fill=False)
        variance_3 = plt.Circle(P2, variances[third_vertex_index], fill=False)

        ax.add_patch(variance_1)
        ax.add_patch(variance_2)
        ax.add_patch(variance_3)
        ax.set_aspect('equal', adjustable='box')  # Equal aspect ratio

    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_title("Centers Triangle")
    plt.savefig(output_file_name)
    # plt.show()
    plt.clf()

def plot_k_vs_accuracy(k_harmonics, mean_accuracies, max_accuracies, mean_output_file_name, max_output_file_name):
    plt.plot(k_harmonics, mean_accuracies)
    plt.title("k vs mean adjusted random score over 10 trials")
    plt.savefig(mean_output_file_name)
    # plt.show()
    plt.clf()

    plt.plot(k_harmonics, max_accuracies)
    plt.title("k vs max adjusted random score over 10 trials")
    plt.savefig(max_output_file_name)
    plt.clf()

def get_purity(clustering_results, true_labels):
    num_correct = 0
    for cluster in range(NUM_CLUSTERS):
        ground_truth = true_labels[clustering_results == cluster]
        majority_label = np.bincount(ground_truth).argmax()
        num_correct += np.count_nonzero(ground_truth == majority_label)

    purity = num_correct / clustering_results.shape[0]
    return purity


# def analysis(input_file, output_directory):
def analysis():
    # print("Now analyzing: ", input_file)
    # # assumption that all vertices are encoded as 0-n consecutive integers
    # f = open(input_file, "r")
    # file_string = f.read()
    # f.close()
    # edges = file_string.split('\n')

    # unordered_g = nx.parse_edgelist(edges, nodetype=int)

    # G = nx.Graph()
    # G.add_nodes_from(sorted(unordered_g.nodes(data=True)))
    # G.add_edges_from(unordered_g.edges(data=True))


    data = load_iris()
    true_clusters = data.target
    points = np.array(data.data)

    adjacency_matrix = kneighbors_graph(points, 50, mode='connectivity', include_self=False)
    G = nx.from_numpy_array(adjacency_matrix)

    # nodes = list(G.nodes)
    # print(nodes)

    if not nx.is_connected(G):
        print("geh")
        quit()

    # true_clusters = get_expected_clustering(nodes)
    mean_accuracies = []
    max_accuracies = []

    for k in K_HARMONICS:
        print("k: ", k)
        accuracy_trials = []
        max_accuracy = -2
        best_triangle = best_normalized_triangle = best_labels = best_centers = best_means = None

        position_encoding = get_position_encoding(k, G)
        expected_centroids = get_ideal_centroids(position_encoding, true_clusters)
        expected_triangle, expected_normal_triangle = get_triangle_lengths(expected_centroids)
        expected_mean_distances, expected_max_distances = calculate_variance(true_clusters, expected_centroids, position_encoding)

        if k==0.1: 
            file_name = "point_1"
        elif k==0.5:
            file_name = "point_5"
        else:
            file_name = str(k)

        for _ in range(10):
            kmeans = return_clustering(G, position_encoding, NUM_CLUSTERS, output_png=file_name)
            accuracy = adjusted_rand_score(true_clusters, kmeans.labels_)
            triangle, normalized_triangle = get_triangle_lengths(kmeans.cluster_centers_)

            accuracy_trials.append(accuracy)

            if (accuracy > max_accuracy):
                best_kmeans = kmeans
                max_accuracy =  accuracy 
                best_labels = kmeans.labels_
                best_centers = kmeans.cluster_centers_
                best_triangle = triangle
                best_normalized_triangle = normalized_triangle

        mean_accuracy = np.mean(accuracy_trials)
        max_accuracy = np.max(accuracy_trials)
        print("Mean accuracy: ", mean_accuracy)
        mean_accuracies.append(mean_accuracy)
        max_accuracies.append(max_accuracy)

        colors = ['red', 'blue', 'green']
        node_colors = [colors[cluster] for cluster in best_kmeans.labels_]

        nx.draw(G, with_labels=True, node_color=node_colors)
        plt.title('Clustered Graph')
        plt.savefig(file_name + "_best_clustering")
        # plt.show()    
        plt.clf()

        max_accuracy_index = accuracy_trials.index(max(accuracy_trials))
        mean_distances, max_distances = calculate_variance(best_labels, best_centers, position_encoding)

        if (accuracy_trials[max_accuracy_index] == 1):
            mean_distances_file_name = file_name + "_mean_dist_(total_accuracy).png"
            normalized_triangle_name = file_name + "_sample_normalized_triangle_(total_accuracy).png"
            max_distances_file_name = file_name + "_max_dist_(total_accuracy).png"
        else:
            mean_distances_file_name = file_name + "_mean_dist.png"
            normalized_triangle_name = file_name + "_sample_normalized_triangle.png"
            max_distances_file_name = file_name + "_max_dist.png"

        ideal_mean_file = file_name + "ideal_triangle_mean.png"
        ideal_max_file = file_name + "ideal_triangle_max.png"
        ideal_normalized = file_name + "ideal_normalized_triangle.png"

        plot_triangle(expected_triangle, ideal_mean_file, variances=expected_mean_distances)
        plot_triangle(expected_triangle, ideal_max_file, variances=expected_max_distances)
        plot_triangle(expected_normal_triangle, ideal_normalized)

        plot_triangle(best_normalized_triangle, normalized_triangle_name)
        plot_triangle(best_triangle, mean_distances_file_name, variances=mean_distances)
        plot_triangle(best_triangle, max_distances_file_name, variances=max_distances)

    mean_k_vs_accuracy_filename = "k_vs_mean_accuracy.png"
    max_k_vs_accuracy_filename = "k_vs_max_accuracy"
    plot_k_vs_accuracy(K_HARMONICS, mean_accuracies, max_accuracies, mean_k_vs_accuracy_filename, max_k_vs_accuracy_filename)

def main():   
    # input_directory = sys.argv[1]
    # output_directory = sys.argv[2]

    # input_files = os.listdir(input_directory)

    # for input_file in input_files:
    #     analysis(os.path.join(input_directory, input_file), output_directory)

    analysis()

if __name__ == "__main__":
    main()
