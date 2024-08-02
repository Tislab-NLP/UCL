import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def contrastive_loss(temp, embedding, label):
    """calculate the contrastive loss
    """
    # print(label)
    # print(label.shape[0])
    # print(embedding.shape[0])
    # cosine similarity between embeddings
    cosine_sim = cosine_similarity(embedding, embedding)
    # remove diagonal elements from matrix
    dis = cosine_sim[~np.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
    # apply temprature to elements
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    # apply exp to elements
    dis = np.exp(dis)
    cosine_sim = np.exp(cosine_sim)

    # calculate row sum
    row_sum = []
    for i in range(len(embedding)):
        row_sum.append(sum(dis[i]))
    # calculate outer sum
    contrastive_loss = 0
    for i in range(len(embedding)):
        n_i = label.tolist().count(label[i]) - 1
        # print(n_i)
        inner_sum = 0
        # calculate inner sum
        for j in range(len(embedding)):
            if label[i] == label[j] and i != j:
                inner_sum  =inner_sum + np.log(cosine_sim[i][j] /row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
        else:
            contrastive_loss += 0
    return contrastive_loss


def F_contrastive_loss(temp, embedding, label,s_label):
    """calculate the contrastive loss
    """
    cosine_sim = cosine_similarity(embedding, embedding)
    # remove diagonal elements from matrix
    dis = cosine_sim[~np.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
    # apply temprature to elements
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    # apply exp to elements
    dis = np.exp(dis)
    cosine_sim = np.exp(cosine_sim)

    # calculate row sum
    row_sum = []
    for i in range(len(embedding)):
        em_sum = 0
        for j in range(len(embedding)-1):
            if s_label[i] == s_label[j] and label[i] != label[j]:
                em_sum += dis[i][j]
        row_sum.append(em_sum)
    # calculate outer sum
    contrastive_loss = 0
    for i in range(len(embedding)):
        n_i = label.tolist().count(label[i]) - 1
        # print(n_i)
        inner_sum = 0
        # calculate inner sum
        for j in range(len(embedding)):
            if label[i] == label[j] and i != j:
                if(row_sum[i] == 0):
                    inner_sum  =inner_sum + np.log(cosine_sim[i][j] /cosine_sim[i][j]+row_sum[i])
                else:
                    inner_sum = inner_sum + np.log(cosine_sim[i][j] /  row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
        else:
            contrastive_loss += 0
    return contrastive_loss



def Fs_contrastive_loss(temp, embedding, label,s_label):
    """calculate the contrastive loss
    """
    cosine_sim = cosine_similarity(embedding, embedding)
    # remove diagonal elements from matrix
    dis = cosine_sim[~np.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
    # apply temprature to elements
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    # apply exp to elements
    dis = np.exp(dis)
    cosine_sim = np.exp(cosine_sim)

    # calculate row sum
    row_sum = []
    for i in range(len(embedding)):
        em_sum = 0
        for j in range(len(embedding)-1):
            if s_label[i] == s_label[j] and label[i] != label[j]:
                em_sum += dis[i][j]
        row_sum.append(em_sum)
    # calculate outer sum
    contrastive_loss = 0
    for i in range(len(embedding)):
        n_i = label.tolist().count(label[i])
        # print(n_i)
        inner_sum = 0
        # calculate inner sum
        for j in range(len(embedding)):
            if label[i] == label[j] and i != j and s_label[i] == s_label[j]:
                if(row_sum[i] == 0):
                    inner_sum  =inner_sum + np.log(cosine_sim[i][j] /cosine_sim[i][j]+row_sum[i])
                else:
                    inner_sum = inner_sum + np.log(cosine_sim[i][j] /  row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
        else:
            contrastive_loss += 0
    return contrastive_loss


