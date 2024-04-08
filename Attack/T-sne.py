import numpy as np
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import pandas as pd


y_true_Mnist = [79.54,124.77,181.74,249.02,266.07,292.44,371.79]
#Mnist: Mnist,Emnist,Fashion,LFW,C100,C10,Emotion,Clothing

y_true_Emnist = [0.26,79.6,118.66,163.93]
#Emnist: Emnist,Mnist,Fashion,LFW,C100,C10,Emotion,Clothing

y_true_Fashion = [0.53,108.72,118.66,124.77]
#Fashion: Fashion,LFW,Emnist,Mnist,C100,C10,Emotion,Clothing

y_true_Clothing = [0.27,107.48,161.09,162.46]

#Clothing:Clothing,Emotion,Cifar10,Cifar100,LFW,Fashion,Emnist,Mnist

y_true_LFW = [0.27,108.72,129.8,130.43]

#LFW:LFW,Fahsion,Emotion,Cifar10,Cifar100,Emnist,Mnist,Clothing

y_true_Emotion = [2.97,97.86,102.33,107.48]

#Emotion:Emotion, Cifar10,Cifar100,Clothing,LFW,Fashion,Emnist,Mnist

y_true_Cifar10 = [0.6,49.66,97.86,130.43]

#Cifar10: Cifar10,Cifar100,Emotion,LFW,Clothing,Fashion,Emnist,Mnist

y_true_Cifar100 = [0.26,49.69,102.33,130.47]

#Cifar100:Cifar100,Cifar10,Emotion,LFW,CLothing,Fashion,Emnist,Mnist


y_pred_Mnist_MI=[0.314,0.242,0.198,0.130,0.124,0.120,10]

y_pred_Emnist_MI=[139,143,138,189]

y_pred_Fashion_MI=[104,134,136,158]

y_pred_Clothing_MI=[207,141,204,212]

y_pred_LFW_MI=[157,127,148,174]

y_pred_Emotion_MI=[148,120,132,123]

y_pred_Cifar10_MI=[149,153,124,161]

y_pred_Cifar100_MI=[155,162,146,175]


        
y_pred_Mnist_o=[11,25,41,74]

y_pred_Emnist_o=[6,21,47,59]

y_pred_Fashion_o=[4,48,58,53]

y_pred_Clothing_o=[21,147,101,131]

y_pred_LFW_o=[5,57,33,78]

y_pred_Emotion_o=[18,129,102,72]

y_pred_Cifar10_o=[59,33,122,37]

y_pred_Cifar100_o=[75,93,93,107]



def get_ndcg(rel_true, rel_pred, p=None, form="linear"):
    """ Returns normalized Discounted Cumulative Gain
    Args:
        rel_true (1-D Array): relevance lists for particular user, (n_songs,)
        rel_pred (1-D Array): predicted relevance lists, (n_pred,)
        p (int): particular rank position
        form (string): two types of nDCG formula, 'linear' or 'exponential'
    Returns:
        ndcg (float): normalized discounted cumulative gain score [0, 1]
    """
    rel_true = np.sort(rel_true)[::-1]
    p = min(len(rel_true), min(len(rel_pred), p))
    discount = 1 / (np.log2(np.arange(p) + 2))

    if form == "linear":
        idcg = np.sum(rel_true[:p] * discount)
        dcg = np.sum(rel_pred[:p] * discount)
    elif form == "exponential" or form == "exp":
        idcg = np.sum([2**x - 1 for x in rel_true[:p]] * discount)
        dcg = np.sum([2**x - 1 for x in rel_pred[:p]] * discount)
    else:
        raise ValueError("Only supported for two formula, 'linear' or 'exp'")

    return dcg / idcg


    
def ndcg(y_true,y_pred):
    return get_ndcg(300-np.array(y_true), 300-np.array(y_pred) ,p=1)
    #return ndcg_score(300-np.array([y_true]), 300-np.array([y_pred]),k=2)

Mnist_MI = ndcg(y_true_Mnist,y_pred_Mnist_MI)

Emnist_MI = ndcg(y_true_Emnist,y_pred_Emnist_MI)

Fashion_MI = ndcg(y_true_Fashion,y_pred_Fashion_MI)

Clothing_MI = ndcg(y_true_Clothing,y_pred_Clothing_MI)

LFW_MI = ndcg(y_true_LFW,y_pred_LFW_MI)

Emotion_MI = ndcg(y_true_Emotion,y_pred_Emotion_MI)

Cifar10_MI = ndcg(y_true_Cifar10,y_pred_Cifar10_MI)

cifar100_MI = ndcg(y_true_Cifar100,y_pred_Cifar100_MI)

Mnist_o = ndcg(y_true_Mnist,y_pred_Mnist_o)

Emnist_o = ndcg(y_true_Emnist,y_pred_Emnist_o)

Fashion_o = ndcg(y_true_Fashion,y_pred_Fashion_o)

Clothing_o = ndcg(y_true_Clothing,y_pred_Clothing_o)

LFW_o = ndcg(y_true_LFW,y_pred_LFW_o)

Emotion_o = ndcg(y_true_Emotion,y_pred_Emotion_o)

Cifar10_o = ndcg(y_true_Cifar10,y_pred_Cifar10_o)

cifar100_o = ndcg(y_true_Cifar100,y_pred_Cifar100_o)




print([Mnist_MI,Emnist_MI,Fashion_MI,Clothing_MI,LFW_MI,Emotion_MI,Cifar10_MI,cifar100_MI])


print([Mnist_o,Emnist_o,Fashion_o,Clothing_o,LFW_o,Emotion_o,Cifar10_o,cifar100_o])
        


            

        
