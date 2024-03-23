import csv

# This file transforms the data provided in knn_data.csv into a dictionary organzied by Patient ID
# for easier lookup and extraction. It also transforms the sequences into their numericl form.


MODE_TO_INT = {
    '*' : 0,
    'S' : 1,
    'L' : 2,
    'M' : 3,
    'V' : 4
}

# will be ignoring the #N/A sequences for now (entry 220)
def numerical_transform(sequence): #TODO: test it
    if sequence == "#N/A":
        return None
    else:
        enc_seq = list(map(lambda x: MODE_TO_INT[x], list(sequence)))
        
    return enc_seq

# create new dictionary of sequences
knn_data = {}

# read in csv file ("convert to a dataframe"?) 
with open("knn_data.csv", mode = 'r') as file:

    # reading file
    data = csv.DictReader(file)

    # line by line
    for lines in data:

        # get entry id
        entry = lines.get("ID")

        # get corresponding patient id
        items = entry.split('_')    # split string by underscores
        patient_id = items[1]   # id is the middle item

        seq = lines.get("Sequence")  # access the value for sequence via .get()

        # convert seq using numerical transform
        encoded_seq = numerical_transform(seq)
        
        # if the patient doesn't already exist, create the list value with seq as the first sequence.
        if patient_id not in knn_data.keys():
            knn_data[patient_id] = [encoded_seq]
        # otherwise, add seq to this patient's sequences
        else:
            knn_data[patient_id].append(encoded_seq)

print(knn_data)


    