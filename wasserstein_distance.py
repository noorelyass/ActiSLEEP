import numpy as np
import ot


def get_wasserstein_distance(i: list[str], j: list[str]): # parameters will be (patient_id: str, sequence: int)
    """
    Given the unique id of a patient <patient_id>, and a 48-digit sequence of numbers between 0 and 4
    representing a sequence of physical activity levels (0 being sleep, 1 being sedementary,
    2 being light, 3 being moderate, and 4 being vigorous) <sequence>, determine the similarity between
    <sequence> and the patient's actual data by returning the Wasserstein distance between the 
    two measures as histograms.

    """
    histogram1 = np.array(i)
    histogram2 = np.array(j)

    # get cost matrix (no normalizing, so it works for both balanced and unbalanced):
    cost_matrix = ot.dist(histogram1.reshape((len(i), 1)), histogram2.reshape((len(j), 1)))
    # print(cost_matrix)

    # get the optimal transport plan matrix:
    gamma =  ot.sinkhorn_unbalanced(histogram1, histogram2, cost_matrix, 0.1, 0.1) 

    # sum up OT plan multiplied by the cost matrix to get a numerical value for WD:
    w_distance = np.sum(gamma * cost_matrix)
    print("Wasserstein Distance: ", w_distance)

    return(w_distance)