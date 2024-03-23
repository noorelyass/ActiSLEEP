from transform_knn import knn_data


def get_patient_data(id: int):
    """ Given the id of a patient, extract and return the patient's data entries.

    """
    # print(id)
    # print(knn_data[id])
    return knn_data[id]

