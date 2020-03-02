import numpy as np
import random

def masking(tokens, p = 0.1, mask='[MASK]'):
    outputs = tokens[:]
    for i in range(len(tokens)):
        if np.random.rand() < p:
            outputs[i] = mask 
    return outputs

def deleting(tokens, p = 0.1):
    choice = np.random.binomial(1,1-p,len(tokens))
    outputs = [tokens[i] for i in range(len(tokens)) if choice[i]==1]
    return outputs


def n_gram_sampling(tokens, 
                    p_ng = [0.2,0.2,0.2,0.2,0.2],
                    l_ng = [1,2,3,4,5]):
    
    span_length = np.random.choice(l_ng,p= p_ng)
    start_position = max(0,np.random.randint(0,len(tokens)-span_length+1))
    n_gram_span = tokens[start_position:start_position+span_length]
    return n_gram_span


def short_disorder(tokens, p = [0.9,0.1,0,0,0]):  # untouched + four cases abc, bac, cba, cab, bca
    i = 0
    outputs = tokens[:]
    l = len(tokens)
    while i < l-1:
        permutation = np.random.choice([0,1,2,3,4],p=p)
        if permutation!=0 and i==l-2:
            outputs[i], outputs[i+1] = outputs[i+1], outputs[i]
            i += 2
        elif permutation==1:
            outputs[i], outputs[i+1] = outputs[i+1], outputs[i]
            i += 2
        elif permutation==2:
            outputs[i], outputs[i+2] = outputs[i+2], outputs[i]
            i +=3
        elif permutation==3:
            outputs[i],outputs[i+1],outputs[i+2] = outputs[i+2],outputs[i],outputs[i+1]
            i += 3
        elif permutation==4:
            outputs[i],outputs[i+1],outputs[i+2] = outputs[i+1],outputs[i+2],outputs[i]
            i += 3
        else:
            i += 1
    return outputs

def long_disorder(tokens,p = 0.1, length=20):
    outputs = tokens[:]
    if int(length) <= 1:
        length = len(tokens)*length
    length = (int(length)+1) //2 * 2
    i = 0
    while i<=len(outputs)-length:
        if np.random.rand() < p:
            outputs[i:i+length//2], outputs[i+length//2:i+length] = outputs[i+length//2:i+length], outputs[i:i+length//2]
            i += length
        else:
            i += 1
    return outputs