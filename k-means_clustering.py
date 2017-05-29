#!/usr/bin/python2
from __future__ import division
import csv
import numpy as np
import math
import random


# if you end up with an empty cluster, skip over it in mean square calculations
# also, consider choosing random element to be the starting positions

class KMeans:
    def __init__(self, training_data, testing_data, class_index, number_of_classes, k, clustering_attempts):
        self.training_data = training_data
        self.number_of_elements = self.training_data.shape[0]
        self.testing_data = testing_data
        self.class_index = class_index
        self.number_of_classes = number_of_classes
        self.k = k
        self.clustering_attempts = clustering_attempts
        self.avg_mse = 99999999
        self.centroids = None
        self.training_clusters = None
        self.testing_clusters = None
        self.avg_mss = None
        self.avg_entropy = None
        self.labels = None
        self.train()
        self.confusion_matrix = self.test()
        self.accuracy = self.get_accuracy()

    def train(self):
        # Generate some clusterings and pick out the best one
        for clustering_attempt in range(self.clustering_attempts):
            print "clustering attempt: " + str(clustering_attempt)
            new_avg_mse, new_centroids, cluster_memberships = self.make_clusters()
            if new_avg_mse < self.avg_mse:
                self.avg_mse = new_avg_mse
                self.centroids = new_centroids
                self.training_clusters = cluster_memberships
        self.avg_mss = self.get_mean_squared_separation()
        self.avg_entropy = self.get_mean_clustering_entropy()
        self.labels = self.get_labels()

    def get_centroids(self):
        """
        Gets random elements from the training set (with the target class removed)
        to use as inital centroids
        :return: K inital centroids
        """
        centroids = []
        num_elements = training_data.shape[0] -1
        for i in range(self.k):
            random_index = random.randint(0, num_elements - 1)
            centroids.append(training_data[random_index, : self.class_index])
        return centroids

    def get_cluster_memberships(self, data, centroids):
        cluster_memberships = [[] for x in range(k)]
        for element_index in range(data.shape[0]):
            element = data[element_index]
            distances = []
            for index in range(k):
                distances.append(self.get_distance(element, centroids[index]))
            smallest_index = distances.index(min(distances))
            cluster_memberships[smallest_index].append(element)
        return cluster_memberships

    @staticmethod
    def get_distance( element1, element2):
        # The target class is sliced off when computing distance
        distance = sum((element1[:64] - element2[:64]) ** 2)
        return distance

    @staticmethod
    def update_centroids(cluster_memberships):
        new_centroids = []
        for membership in cluster_memberships:
            new_centroids.append(sum(membership)/len(membership))
        return new_centroids

    @staticmethod
    def equal_centroids(old_centroids, new_centroids):
        if not old_centroids:
            return False
        for centroid_index in range(len(new_centroids)):
            for feature_index in range(len(new_centroids[centroid_index])):
                if old_centroids[centroid_index][feature_index] != new_centroids[centroid_index][feature_index]:
                    return False

        return True

    def avg_means_squared_error(self, centroids, cluster_memberships):
        """
        This function returns the average distance between all elements
        and their clusters centroid.
        :param centroids: The cluster centroids
        :param cluster_memberships: A list of elements that belong to each cluster
        :return: 
        """
        number_of_centroids = len(centroids)
        total_distance = 0
        for index in range(number_of_centroids):
            total_distance += self.means_squared_error(centroids[index], cluster_memberships[index])
        return total_distance / number_of_centroids

    def means_squared_error(self, centroid, membership):
        """
        This function returns the average distance in a cluster between the
        cluster elements and the cluster centroid.
        It does this by summing the distances between each element in the 
        cluster and the centroid and dividing that by the number of elements
        :param centroid: The cluster centroid
        :param membership: A list of elements that belong to this cluster
        :return: the average distance between elements and the cluster centroid
        """

        total_distance = 0
        for member in membership:
            total_distance += self.get_distance(member, centroid)
        return total_distance / len(membership)

    def get_mean_squared_separation(self):
        """
        Gets the average distance between all the centroids
        :return: The average distance between the centroids
        """

        total_distance = 0
        for centroid1 in range(self.k):
            for centroid2 in range(self.k):
                if centroid1 != centroid2:
                    total_distance += self.get_distance(self.centroids[centroid1], self.centroids[centroid2])
        return total_distance / ((self.k * self.k - 1) / 2)

    def get_mean_clustering_entropy(self):
        """
         Returns the mean entropy of the clustering
        """
        total = 0
        for i in range(self.k):
            total += len(self.training_clusters[i]) / self.number_of_elements * self.get_mean_cluster_entropy(self.training_clusters[i])
        return total

    def get_mean_cluster_entropy(self, membership):
        member_class_counts = self.get_cluster_class_counts(membership)
        total = 0
        for class_count in member_class_counts:
            class_fraction = class_count/len(membership)
            if class_fraction != 0:
                total += class_fraction * math.log(class_fraction, 2)
        return - total

    def get_labels(self):
        """
        Assigns a class label to each cluster.
        The cluster with the most instances of a class gets that
        label.
        Ties are broken randomly.
        :return: a list of labels where the position of that label 
        corresponds to which cluster it's labeling
        """
        labels = []
        for cluster_index in range(k):
            labels.append(self.get_label(self.training_clusters[cluster_index]))
        return labels

    def get_label(self, cluster):
        """
        Gets the most common class in the cluster for the label.
        In the case of ties, they are broken randomly
        :param cluster: The cluster to label 
        :return: 
        """
        class_counts = self.get_cluster_class_counts(cluster)
        # Find the most common class
        max_val = max(class_counts)
        # Get any other classes that are as common
        max_index = [x for x in range(len(class_counts)) if class_counts[x] == max_val]
        if len(max_index) > 1:
            random.shuffle(max_index)
        return max_index[0]

    def test(self):
        confusion_matrix = np.zeros((self.k, self.k))
        # Assign all test data to clusters
        self.testing_clusters = self.get_cluster_memberships(self.testing_data, self.centroids)
        # For each cluster
        for membership_index in range(self.k):
            # For each test member of that cluster
            for member_index in range(len(self.testing_clusters[membership_index])):
                # Enter the predicted and target class into the confusion matrix
                actual_class = self.testing_clusters[membership_index][member_index][self.class_index:][0]
                predicted_class = self.labels[membership_index]
                confusion_matrix[predicted_class][actual_class] += 1
        return confusion_matrix

    def get_accuracy(self):
        correct = sum(np.diagonal(self.confusion_matrix))
        total = len(self.testing_data)
        return correct / total

    def get_cluster_class_counts(self, cluster):
        """
        Gets the number of times a class is in a cluster
        :param cluster: the elements in the cluster
        :return: A list of the class counts in the cluster
        """
        class_counts = [0 for x in range(self.number_of_classes)]
        for member in cluster:
            class_counts[member[64]] += 1
        return class_counts

    def make_clusters(self):
        """
        This function will make clusters by selecting random
        elements as starting centroids and then finding clusters
        based on that and shifting the centroids to be better
        centered until the centroids no longer move.
        It returns the average distance between all elements
        and their centroids, the centroids themselves, and the 
        memberships of each element to each cluster
        :return: avg_mse, centroids, memberships
        """
        old_centroids = []
        centroids = self.get_centroids()

        recenter_count = 0
        while not self.equal_centroids(old_centroids, centroids):
            print "recentered " + str(recenter_count) + " times"
            cluster_memberships = self.get_cluster_memberships(self.training_data, centroids)
            old_centroids = centroids
            centroids = self.update_centroids(cluster_memberships)
            recenter_count += 1

        return self.avg_means_squared_error(centroids, cluster_memberships), centroids, cluster_memberships

    def save_superimposed_clusters(self):
        for cluster_index in range(len(self.testing_clusters)):
            self.save_superimposed_cluster(cluster_index)

    def save_superimposed_cluster(self, index):
        # superimpose the cluster members
        superimposed_cluster = sum(self.testing_clusters[index])
        # strip off the class information
        superimposed_cluster = superimposed_cluster[:self.class_index]
        superimposed_cluster.shape = (8, 8)
        out_file = open("superimposed_data/superimposed_" + str(index), 'w')
        #np.savetxt(out_file, superimposed_cluster, delimiter=',')
        np.save(out_file, superimposed_cluster)
        out_file.close


def read_data(filename):
    """Massages the data from csv into numpy arrays"""
    print("Loading " + filename)
    data = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    data = np.array(data)
    # convert from string to int
    data = data.astype(np.int64)
    print("Done")
    return data


if __name__ == "__main__":
    # read in data
    training_data = read_data("./optdigits/optdigits.train")
    testing_data = read_data("./optdigits/optdigits.test")

    k = 10
    cluster_attempts = 5
    class_index = 64
    number_of_classes = 10

    kMeans = KMeans(training_data, testing_data, class_index, number_of_classes, k, cluster_attempts)
    kMeans.save_superimposed_clusters()
    print "Accuracy: " + str(kMeans.accuracy)
    print "Confusion Matrix:"
    print kMeans.confusion_matrix

#TODO: put back into dry run testing to spit out image files, get that working before you try the whole thing