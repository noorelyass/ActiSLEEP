import numpy as np
import matplotlib.pyplot as pl
import ot
import ot.plot
from transform_knn import encoded_sequences

# Histogram bars have heights of any of {0, 1, 2, 3, 4}
# Number of bars/entries: 48 -> for every 1/2 hour (30 mins) x 24 hours

# For our data, we are dealing with unbalanced optimal transport - our distributions 
# (histograms) are NOT guaranteed to have the same total mass.

# ----------------------------------------------------------------------------------------------------------------------- #

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
    print(cost_matrix)

    # get the optimal transport plan matrix:
    gamma =  ot.sinkhorn_unbalanced(histogram1, histogram2, cost_matrix, 0.1, 0.1) 

    # sum up OT plan multiplied by the cost matrix to get a numerical value for WD:
    w_distance = np.sum(gamma * cost_matrix)
    print("Wasserstein Distance: ", w_distance)

    # Plot the histograms
    bins = np.arange(len(histogram1))
    width = 0.35

    pl.subplots_adjust(bottom=0.2) # so that Wasserstein Distance is visible.

    pl.bar(bins, histogram1, width, alpha=0.5, label='Source')
    pl.bar(bins + width, histogram2, width, alpha=0.5, label='Target')

    pl.xlabel('Bins \n \n' + "Wasserstein Distance: " + str(w_distance))
    pl.ylabel('Level of Physical Activity')
    pl.title('Histograms')
    pl.xticks(bins + width/2, bins)
    pl.legend()

    return(w_distance)

    # Transport plot:
    # pl.figure(4, figsize=(6.4, 3))
    # pl.bar(bins, histogram1, width, label='Source distribution')
    # pl.bar(bins, histogram2, width, label='Target distribution')
    # pl.fill(bins, gamma.sum(1), 'b', alpha=0.5, label='Transported source')
    # pl.fill(bins, gamma.sum(0), 'r', alpha=0.5, label='Transported target')
    # pl.legend(loc='upper right')
    # pl.title('Distributions and transported mass for UOT')

# ------------------------------------------------------------------------------------------------------------------ #

if __name__ == "__main__":
    # take first ten sequences (so 5 histograms will be generated)
    ten_seq = encoded_sequences[:10]

    # computing wasserstein distance pairwise:
    for i, j in zip(ten_seq[0::2], ten_seq[1::2]):
        print(i, j)
        get_wasserstein_distance(i, j)
        pl.show()

    # sample cases:

    # target == source:
    # test = [[1,1,1,1,1], [1,1,1,1,1]]           # should give us 0.

    # small difference:
    # test = [[1,2,3,4,3], [2,3,4,3,2]]            # should give a relatively small dist.

    # larger difference:
    # test = [[1,2,1,0,2], [5,7,9,6,5]]         # should give relatively larger dist. than ^

    # target is double source:
    # test = [[0,0,1,1,1,1,1,3,1,1], [0,0,2,2,2,2,2,6,2,2]]

    # source is double target:
    # test = [[0,0,2,2,2,2,2,6,2,2], [0,0,1,1,1,1,1,3,1,1]]

    # for i, j in zip(test[0::2], test[1::2]):
    #     print(i, j)
    #     get_wasserstein_distance(i, j)
    #     pl.show()

 # ------------------------------------------------------------------------------------------------ #
