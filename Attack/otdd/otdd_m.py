from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance
from utils import *
from torchvision import datasets, transforms
from torch.utils import data
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd  
np.set_printoptions(suppress=True)

class Otdd_measure:

    def __init__(self,max_sample=2000, class_num=5, name_list=['LFW','Emotion','Emnist','Cifar10','Cifar100','Mnist','Clothing','Fashion'], Data_addr='./Data_train/'):

        loaders0 = [] 

        for name in name_list:

            loaders0.append(init_dataloader(name,Data_addr+name+'/train/label.txt',max_size=max_sample,class_num = class_num))

        self.Loaders = []

        self.name_list = name_list
        
        for i in loaders0:

            temp = i

            temp.dataset.targets = torch.tensor(temp.dataset.targets)

            self.Loaders.append(temp)

        self.dist = []

        self.get_distance()

        self.dist = np.around(np.array(self.dist),2)
            

    def get_distance(self):


        for i in range(len(self.Loaders)):

            Loader1 = self.Loaders[i]

            self.dist.append([])

            for j in range(len(self.Loaders)):

                Loader2 = self.Loaders[j]
                
                temp_dist = DatasetDistance(Loader1, Loader2,
                       inner_ot_method = 'exact',
                       debiased_loss = True,
                       p = 2, entreg = 1e-1,
                       device='cpu')
                
                d = temp_dist.distance(maxsamples = 1000)

                self.dist[i].append(d)
                
    def draw(self):
        
        corr = self.dist

        x_axis = self.name_list

        y_axis = self.name_list
            
        ax = sns.heatmap(corr, annot=True,cmap="YlGnBu",xticklabels=x_axis, yticklabels=y_axis, square=True,fmt='g')

        plt.show()

    def draw_t_sne(self):
        df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)

if __name__ == '__main__':

    name_list = ['LFW','Emotion','Emnist']

    o = Otdd_measure()

    o.draw()

    print(o.list)


