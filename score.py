from patient_data import get_patient_data
from wasserstein_distance import get_wasserstein_distance

def get_score(id: str, sequence: any) -> float:
    """ Given a patient ID <id> and a sequence <sequence>, determine and return an error score
    for <sequence> by computing the Wasserstein distance between <sequence> and each sequence
    found in the patient's data.

    The error score is calculated summing up the squared Wasserstein distances between <sequence>
    and each sequence in the patient's data, and then dividing by the total number of
    sequences in get_patient_data[id] for an average.

    Precondition: len(sequence) == len(get_patient_data[id][i]) for i in get_patient_data[id].

    """
    training_data = get_patient_data(id)

    sum = 0

    for i in range(len(training_data)):
        w_dis = get_wasserstein_distance(training_data[i], sequence)
        sum += w_dis

    score = sum / len(training_data)

    return score