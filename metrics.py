import numpy as np

def FPR(labels,probs):

    ###initialize
    N = 0
    for i in labels:
        if (i == 0):
            N += 1
    FP = 0
    i = 0
    while (i  < len(probs) ):
        if (probs[i] == 1 and labels[i] == 0):
            FP += 1
        i += 1
    FPR = FP / N
    #print(FP,N)
    return FPR

def FNR(labels,probs):
    ###initialize
    P = 0
    for i in labels:
        if (i == 1):
            P += 1
    FN = 0
    TN = 0
    i = 0
    while (i  < len(probs) ):

        if (probs[i] == 0 and labels[i] == 1):

            FN += 1
        i += 1
    FNR = FN / P
    #print(FN,P)
    return FNR

def FPR_m(labels,probs,s_labels):

    ###initialize
    N = 0
    i = 0
    while (i<len(labels)):
        if(labels[i] == 0 and s_labels[i] == 0):
            N+=1
        i+=1
    FP = 0
    i = 0
    while (i  < len(probs) ):
        if (probs[i] == 1 and labels[i] == 0 and s_labels[i] == 0):
            FP += 1
        i += 1
    FPR_m = FP / N
    #print(FP,N,'js')
    return FPR_m


def FNR_m(labels,probs,s_labels):
    ###initialize
    P = 0
    i = 0
    while (i<len(labels)):
        if(labels[i] == 1 and s_labels[i] == 0):
            P+=1
        i+=1
    FN = 0
    i = 0
    while (i  < len(probs) ):

        if (probs[i] == 0 and labels[i] == 1 and s_labels[i] == 0):

            FN += 1
        i += 1
    FNR_m = FN / P
    #print(FN,P)
    return FNR_m

def FPR_f(labels,probs,s_labels):

    ###initialize
    N = 0
    i = 0
    while (i<len(labels)):
        if(labels[i] == 0 and s_labels[i] == 1):
            N+=1
        i+=1
    FP = 0
    i = 0
    while (i  < len(probs) ):
        if (probs[i] == 1 and labels[i] == 0 and s_labels[i] == 1):
            FP += 1
        i += 1
    FPR_f = FP / N
    #print(FP,N)
    return FPR_f

def FNR_f(labels,probs,s_labels):
    ###initialize
    P = 0
    i = 0
    while (i<len(labels)):
        if(labels[i] == 1 and s_labels[i] == 1):
            P+=1
        i+=1
    FN = 0
    i = 0
    while (i  < len(probs) ):

        if (probs[i] == 0 and labels[i] == 1 and s_labels[i] == 1):

            FN += 1
        i += 1
    FNR_f = FN / P
    #print(FN,P,'JS')
    return FNR_f

def FPED_B (FPR_m1,FPR_f1,FPR1):
    FPED_b = abs(FPR_m1 - FPR1) + abs(FPR_f1 - FPR1)
    return FPED_b
def FNED_B (FNR_m1,FNR_f1,FNR1):
    FNED_b = abs(FNR_m1 - FNR1)+ abs(FNR_f1 - FNR1)
    return FNED_b

def FPED_P (FPR_m1,FPR_f1):
    FPED_p = abs(FPR_m1 - FPR_f1)
    return FPED_p
def FNED_P (FNR_m1,FNR_f1):
    FNED_p = abs(FNR_m1 - FNR_f1)
    return FNED_p

