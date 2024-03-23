import numpy as np

def reduce_slots(sequence, slots: int):
    """ This function returns a <slots> version of the sequence passed in.

    This version of the sequence is detailed by interval points x1, x2, ..., until x_<slots>-1 with
    values between 1 and 23 marking the slot duration, and heights of each slot h1, h2, 
    ..., h<slots> with values between 0 and 4 representing the physical activity levels.
    """

    # since this function is only used once in get_optimal_sequence() to simplify our S_0, 
    # we pre-set the intervals as follows to give use equal durations:
    # could be generalized as (len(sequence)/4) - 1, increment amount each time
    x1 = 5
    x2 = 11
    x3 = 17

    # we then average the heights of <sequence> between 0, x1, x2, x3, and the end for our height values.
    h1 = round(np.average(sequence[:x1]))
    h2 = round(np.average(sequence[x1:x2]))
    h3 = round(np.average(sequence[x2:x3]))
    h4 = round(np.average(sequence[x3:]))

    # return the xs and hs in a tuple.
    print((x1, x2, x3), (h1, h2, h3, h4))
    return ((x1, x2, x3), (h1, h2, h3, h4))




