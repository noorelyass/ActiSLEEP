import numpy as np
import matplotlib.pylab as pl
import ot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

from transform_knn import encoded_sequences

def wasserstein_barycenter_2d(s: list[list[int]], round=False):
    """ Given an list of lists of histogram bin values (integers) <s>, calculate the Wasserstein barycenter,
    and generate a 2D visualization showing the histograms as well as the Wassertsein barycenter overtop.

    If <round> is True, generate a barycenter with bins rounded to the nearest whole number.

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
    
    # arrange the bin positions
    x = np.arange(nbins, dtype=np.float64)

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

    ####################################### 2D PLOTTING ##################################

    pl.figure(1, figsize=(10, 6))
    
    for i in range(len(hgs)):
        pl.bar(x, hgs[i], width=1.0, alpha=0.5, label=('Sequence ' + str(i)), edgecolor='white')

    # rounded case
    if round:
        rounded_bary = (np.rint(rescaled_bary)).astype(int)      # rounds each bin to the nearest whole value
        pl.bar(x, rounded_bary, width=1.0, label='Wasserstein', alpha=0.5, color='red', edgecolor='darkred')

        pl.title("Distributions and Wasserstein Barycenter - Rounded")
        wb_sequence = "Wasserstein Barycenter as a sequence: " + str(rounded_bary)

        pl.subplots_adjust(top=0.85, bottom=0.15)

        pl.figtext(0.05, 0.05, wb_sequence, fontsize=10,
            verticalalignment='bottom', bbox=dict(facecolor='lightblue', edgecolor='black'))
        
    else:
        pl.bar(x, rescaled_bary, width=1.0, label='Wasserstein', alpha=0.5, color='red', edgecolor='darkred')
        
        pl.title("Distributions and Wasserstein Barycenter")
        wb_sequence = "Wasserstein Barycenter as a sequence: " + str(rescaled_bary)
        
    pl.xlabel('Hour')
    pl.ylabel('Level of Physical Activity')
    pl.legend()

    r = "reg: " + str(reg) + "\nreg_m: " + str(reg_m)

    pl.figtext(0.05, 0.95, r, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='yellow', edgecolor='black'))

    pl.show()

##############################################################################################################################################

if __name__ == "__main__":
    # balanced
    # wasserstein_barycenter_2d([[1,2,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,2,1]])
    # wasserstein_barycenter_2d([[1,2,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,2,1]], True)       # rounded
    

    # unbalanced
    wasserstein_barycenter_2d([[12,10,9,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,2,1]])
    # wasserstein_barycenter_2d([[3,8,3,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,2,1]], True)       # rounded
    

    # left skewed
    wasserstein_barycenter_2d([[1,2,1,0,0,0,0,0,0,0], [4,2,9,2,0,0,0,0,0,0]])

    # right skewed
    # wasserstein_barycenter_2d([[0,0,0,0,0,0,4,8,12,1], [0,0,0,0,0,0,4,2,9,12]])
    
    # two-peak
    # wasserstein_barycenter_2d([[0,0,0,0,0,0,4,8,12,1], [4,2,9,12,0,0,0,0,0,0]])
    

    # first 10 knn sequences
    # wasserstein_barycenter_2d(encoded_sequences[:10]) 
    # wasserstein_barycenter_2d(encoded_sequences[:10], True)       # rounded

