import numpy as np
import matplotlib.pylab as pl
import ot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

def wasserstein_barycenter(s: list[list[int]], round=True):
    """ Given an list of lists of histogram bin values (integers) <s>, calculate and return
    the Wasserstein barycenter. This will be an array.

    If <round> is True, generate a barycenter with bins rounded to the nearest whole number.
    <round> is defaulted to True.

    Precondition: Each list in <s> must be of the same length.
    """
    ####################################### HISTOGRAMS ##################################

    # convert each list in s into a histogram
    hgs = []                # our list of histograms (plot)
    rescaled_hgs = []       # normalized histograms (calculations)
    s_weights = []          # saving the weights of each histogram for later (barycenter rescaling)


    for sequence in s:
        histogram = np.array(sequence)
        weight = np.sum(histogram)
        rescaled_histogram = histogram / weight

        hgs.append(histogram)
        s_weights.append(weight)
        rescaled_hgs.append(rescaled_histogram)

    # get sum of all of the weights
    sum_weights = np.sum(np.array(s_weights))
    s_weights = np.array(s_weights) / sum_weights

    ####################################### PARAMETERS ##################################
    
    # get number of bins
    nbins = len(s[0])       # using first sequence's length to determine this
    
    # create a matrix A that contains all of the distributions
    A = np.vstack(rescaled_hgs).T

    # number of sequences
    num_hgs = A.shape[1]        # could also just do len(s) or len(hgs)

    # create a cost/loss matrix
    cost_matrix = ot.utils.dist0(nbins)
    cost_matrix /= cost_matrix.max()        # normalizing

    # set regularization parameters
    reg = 0.01
    reg_m = 0.01
    
    ####################################### BARYCENTER ##################################

    # bary_wasserstein = ot.barycenter(A, cost_matrix, reg, weights=s_weights)

    bary_wasserstein = ot.barycenter_unbalanced(A, cost_matrix, reg, reg_m) # , # weights=s_weights)

    rescaled_bary = (bary_wasserstein * sum_weights) / num_hgs
    # ^ this allows takes the weighted average, allowing us to factor in varying masses.

    if round:
        rounded_bary = np.round(rescaled_bary)
        print("Rounded Wasserstein Barycenter: " + str(rounded_bary))
        return rounded_bary
    
    else:
        print("Wasserstein Barycenter: " + str(rescaled_bary))
        return rescaled_bary
    