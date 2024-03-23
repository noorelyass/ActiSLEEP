import numpy as np

""" By quanitzing, we mean represesnting the sequence with k-1 + k values (if k = 4):

x_1, x_2, ... , x_(k-1) 

--> these are the intervals dividing our slots, hence k - 1.

AND

v_1, v_2, ... , v_(k)

--> these are the heights of each of the k slots.

"""

def quantize(reduced_sequence):
    """ This function takes a reduced sequence (AKA one with parameters for x and h) and returns
    a 24-slot version.
    """
    new_array = []

    x1 = reduced_sequence[0][0]
    x2 = reduced_sequence[0][1]
    x3 = reduced_sequence[0][2]

    h1 = reduced_sequence[1][0]
    h2 = reduced_sequence[1][1]
    h3 = reduced_sequence[1][2]
    h4 = reduced_sequence[1][3]

    for i in range(x1):
        new_array.append(h1)

    for j in range(x2-x1):
        new_array.append(h2)

    for k in range(x3-x2):
        new_array.append(h3)

    for l in range(24-x3):
        new_array.append(h4)

    quantified = np.array(new_array)

    return quantified