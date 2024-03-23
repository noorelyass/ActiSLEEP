from patient_data import get_patient_data
from barycenter import wasserstein_barycenter
from reduce import reduce_slots
from score import get_score
from quantizer import quantize
from optimize import optimize, found_best, within_constraints

# Constraints to remember:

MIN_VALUE = 0
MAX_VALUE = 4

k = 4

""" 
GOAL: return H* -> optimized integer sequence that is best representative of the training data.

Each patient in knn_data has between 1 and 10 24-slot data entries. Entry specified by <id>_<entrynum>.

Input the patient id and generate H*.

How do we generate H*? Follow the steps in task summary!

"""
def get_optimal_sequence(id: str):
    """ Given a patient id <id>, return the optimal integer sequence that is best representative 
    of the training data.

    """
    # fcn takes in patient id and returns this patient's entries.
    # apply transform knn to get sequences
    training_data = get_patient_data(id)

    # use barycenter fcn to get barycenter -> this will be starting sequence S_0
    barycenter = wasserstein_barycenter(training_data)
    S_0 = barycenter

    # use reduce fcn to reduce S_0 into k = 4 slots, in duration form.
    reduced = reduce_slots(S_0, k)
    print("Reduced: ", reduced)
    reduced_quantized = quantize(reduced)
    print("Quantized: ", reduced_quantized)

    # use score fcn to calculate score of S_0 and save as best score to start
    best_score = get_score(id, S_0)
    print("Score: ", best_score)

    # use optimize fcn (look and search) to optimize S_0
    while not found_best:
        optimize(reduced) # recursive fcn
    
    # return S_0 as our H*
    # return S_0

if __name__ == '__main__':
    print("hello")
    # print(wasserstein_barycenter(get_patient_data('0105')))
    get_optimal_sequence('0105')