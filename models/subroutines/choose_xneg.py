# choose random x neg
import random
import numpy as np
# procedure to choose xneg by sampling from other subseries 


other_subseries = [morningdf, earlymorningdf, afternoondf] # when evening_train
other_subseries = [eveningdf, earlymorningdf, afternoondf] # when morning_train
other_subseries = [morningdf, eveningdf, afternoondf] # when earlymorningtrain
other_subseries = [morningdf, earlymorningdf, eveningdf] # when afternoontrain

# choose one subseries at random
number_other = 3

def choose_xneg(alength): # returns array of length alength
    rand_num = random.randint(0, number_other-1)
    target_neg_df = other_subseries[rand_num]
    # sample of length alength
    negarray = np.array(target_neg_df['iso'])
    len_negarray = len(negarray)
    start = np.random.randint(0, len_negarray - alength)
    return negarray[start:start+alength]


