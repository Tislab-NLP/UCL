def FPR_mf(labels,probs,s_labels):

    ###initialize
    N = 0
    i = 0
    while (i<len(labels)):
        if(labels[i] == 0 and s_labels[i] == '2'):
            N+=1
        i+=1
    FP = 0
    i = 0
    while (i  < len(probs) ):
        if (probs[i] == 1 and labels[i] == 0 and s_labels[i] == '2'):
            FP += 1
        i += 1
    FPR_m = FP / N
    #print(FP,N,'js')
    return FPR_m


def FNR_mf(labels,probs,s_labels):
    ###initialize
    P = 0
    i = 0
    while (i<len(labels)):
        if(labels[i] == 1 and s_labels[i] == '2'):
            P+=1
        i+=1
    FN = 0
    i = 0
    while (i  < len(probs) ):

        if (probs[i] == 0 and labels[i] == 1 and s_labels[i] == '2'):

            FN += 1
        i += 1
    FNR_m = FN / P
    #print(FN,P)
    return FNR_m

def FPR_3(labels,probs,s_labels):

    ###initialize
    N = 0
    i = 0
    while (i<len(labels)):
        if(labels[i] == 0 and s_labels[i] == '3'):
            N+=1
        i+=1
    FP = 0
    i = 0
    while (i  < len(probs) ):
        if (probs[i] == 1 and labels[i] == 0 and s_labels[i] == '3'):
            FP += 1
        i += 1
    FPR_f = FP / N
    #print(FP,N)
    return FPR_f

def FNR_3(labels,probs,s_labels):
    ###initialize
    P = 0
    i = 0
    while (i<len(labels)):
        if(labels[i] == 1 and s_labels[i] == '3'):
            P+=1
        i+=1
    FN = 0
    i = 0
    while (i  < len(probs) ):

        if (probs[i] == 0 and labels[i] == 1 and s_labels[i] == '3'):

            FN += 1
        i += 1
    FNR_f = FN / P
    #print(FN,P,'JS')
    return FNR_f

def FPED_B1 (FPR_m1,FPR_f1,FPR_mf,FPR_3,FPR1):
    FPED_b = abs(FPR_m1 - FPR1) + abs(FPR_f1 - FPR1) +abs(FPR_mf - FPR1) + abs(FPR_3 - FPR1)
    return FPED_b
def FNED_B1 (FNR_m1,FNR_f1,FNR_mf,FNR_3,FNR1):
    FNED_b = abs(FNR_m1 - FNR1)+ abs(FNR_f1 - FNR1) +abs(FNR_mf - FNR1) +abs(FNR_3 - FNR1)
    return FNED_b

def FPED_P1 (FPR_m1,FPR_f1,FPR_mf,FPR_3):
    FPED_p = abs(FPR_m1 - FPR_f1) + abs(FPR_m1 - FPR_mf) +abs(FPR_m1 - FPR_3) +abs(FPR_f1 - FPR_mf) +abs(FPR_f1 - FPR_3) +abs(FPR_3 - FPR_mf)
    return FPED_p
def FNED_P1 (FNR_m1,FNR_f1,FNR_mf,FNR_3):
    FNED_p = abs(FNR_m1 - FNR_f1) +abs(FNR_m1 - FNR_mf) +abs(FNR_m1 - FNR_3) +abs(FNR_mf - FNR_f1) +abs(FNR_3 - FNR_f1) +abs(FNR_mf - FNR_3)
    return FNED_p
