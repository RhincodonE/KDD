'''
I implemented feature-based clustering in this file. The images will be clustered

into multiple classes and the final result will be stored in to 2 files and 1 folder

in ./Data/con/original/train
'''




from tensorflow.keras.datasets import cifar10,cifar100,fashion_mnist,mnist
from emnist import extract_train_samples,extract_test_samples
import numpy as np
import PIL
from PIL import Image
import csv
import os
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.datasets import fetch_lfw_people
from os.path import exists
from torchvision import transforms 
from numpy import random
import skimage.measure
import os
import numpy as np
from sklearn.cluster import KMeans
import cv2
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from resnet import ResNet18_f_extract

#Tool functions

def read_image(image_name, script_dir):

    image_path = script_dir+'images/'+image_name

    img = Image.open(image_path)
            
    return np.array(img)

def search_index(elements, target):

    index = np.array([])

    for i in elements:

        temp = np.array(np.where(target == i))

        index = np.hstack((index,temp[0]))
        
    return index.astype(int)
    
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

##########Define dataset loaders#########
class Lfw():

    def __init__(self):

        self.name = 'Lfw'
        
        lfw_people = fetch_lfw_people(min_faces_per_person=53, slice_=(slice(72, 192, None), slice(76, 172, None)))

        self.X = lfw_people.data.reshape((lfw_people.images.shape))

        self.y = lfw_people.target
        
        self.target_names = lfw_people.target_names
        
        self.n_classes = self.target_names.shape[0]

    def load_data(self):

        self.X = np.array([skimage.measure.block_reduce(self.X[i],(2,2),np.mean) for i in range(self.X.shape[0])])

        self.X = np.array([np.pad(self.X[i], ((3,3),(6,6)), 'constant') for i in range(self.X.shape[0])])

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=3)
 
        return (X_train,y_train.reshape((y_train.shape[0]))), (X_test,y_test.reshape((y_test.shape[0])))

class Emotion():

    def __init__(self):

        self.name = 'Emotion'

        full_path_csv =  './Data/Emotion/legend.csv'
        
        ifile  = open(full_path_csv, "r")

        reader = csv.reader(ifile)

        pics = []

        label = []

        for row in reader:
                
            pic_name = row[1]

            temp = read_image(pic_name,'./Data/Emotion/')

            if len(temp.shape)==2:
                
                pics.append(temp)

                if row[2] == 'anger':
                
                    label.append(0)

                elif row[2] == 'surprise':
                
                    label.append(1)

                elif row[2] == 'disgust':
                
                    label.append(2)

                elif row[2] == 'fear':
                
                    label.append(3)
                
                elif row[2] == 'neutral':
                
                    label.append(4)
                
                elif row[2] == 'happiness':
                
                    label.append(5)

                else:
                
                    label.append(6)

            else:

                continue

        ifile.close()

        X_train = np.array(pics)

        X_train = np.array([skimage.measure.block_reduce( X_train[i],(14,14),np.mean) for i in range( X_train.shape[0])])

        print(X_train.shape)
        self.X = np.array([np.pad( X_train[i], ((1,2),(1,2)), 'constant') for i in range( X_train.shape[0])])
        print(self.X.shape)
        self.y = np.array(label)

    def load_data(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=3)
 
        return (X_train,y_train.reshape((y_train.shape[0]))), (X_test,y_test.reshape((y_test.shape[0])))

class Clothing():

    def __init__(self):

        self.name = 'Clothing'

        image_size = (28, 28)
        
        batch_size = 32

        train_gen = ImageDataGenerator(preprocessing_function=None)

        train_ds = train_gen.flow_from_directory("./Data/Clothing",seed=1,target_size=image_size,batch_size=batch_size)

        X=[]

        y=[]

        batches = 0

        for X_batch,y_batch in train_ds:

            X.append(X_batch)

            y.append(y_batch)

            batches += 1
            
            if batches >= 3792/batch_size:
            
                break

        X = np.vstack(X)

        self.X =  np.array([rgb2gray(X[i]) for i in range(X.shape[0])])
                        
        y = np.vstack(y)

        self.y = np.argmax(y,axis=1)


    def load_data(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=3)
        print(X_train.shape)
        return (X_train,y_train.reshape((y_train.shape[0]))), (X_test,y_test.reshape((y_test.shape[0])))

class Emnist():

    def __init__(self):

        self.name = 'Emnist'

        X_train, y_train = extract_train_samples('letters')

        X_test, y_test = extract_test_samples('letters')

        y_train = y_train.reshape((y_train.shape[0]))

        y_test = y_test.reshape((y_test.shape[0]))

        elements = [10,1,2,3,4,5,6,7,8,9]

        ind_train = search_index(elements,y_train)
        
        ind_test = search_index(elements,y_test)
        
        self.X_train =  X_train[ind_train]

        self.y_train = y_train[ind_train]

        self.X_test = X_test[ind_test]

        self.y_test = y_test[ind_test]

        self.y_train[self.y_train == 10] = 0
        
        self.y_test[self.y_test == 10] = 0
        
    def load_data(self):
        
        return (self. X_train,self.y_train.reshape((self.y_train.shape[0]))), (self.X_test,self.y_test.reshape((self.y_test.shape[0])))

class Cifar100():

    def __init__(self):

        self.name = 'Cifar100'

        ( X_train, y_train), (X_test, y_test)= cifar100.load_data()

        y_train = y_train.reshape((y_train.shape[0]))

        y_test = y_test.reshape((y_test.shape[0]))

        X_train = np.array([rgb2gray( X_train[i]) for i in range( X_train.shape[0])])

        X_train = np.array([np.pad( X_train[i], ((2,2),(2,2)), 'constant') for i in range( X_train.shape[0])])

        X_test = np.array([rgb2gray(X_test[i]) for i in range(X_test.shape[0])])

        X_test = np.array([np.pad(X_test[i], ((2,2),(2,2)), 'constant') for i in range(X_test.shape[0])])

        #elements = random.randint(0,99,(10))

        elements = [0,1,2,3,4,5,6,7,8,9]

        ind_train = search_index(elements,y_train)
        
        ind_test = search_index(elements,y_test)

        self. X_train =  X_train[ind_train]

        self.y_train = y_train[ind_train]

        self.X_test = X_test[ind_test]

        self.y_test = y_test[ind_test]

    def load_data(self):
        
        return (self. X_train,self.y_train.reshape((self.y_train.shape[0]))), (self.X_test,self.y_test.reshape((self.y_test.shape[0])))

class Cifar10():

    def __init__(self):

        self.name = 'Cifar10'

        (self.X_train, self.y_train), (self.X_test, self.y_test)= cifar10.load_data()

        self.y_train = self.y_train.reshape((self.y_train.shape[0]))

        self.y_test = self.y_test.reshape((self.y_test.shape[0]))

        X_train = np.array([rgb2gray( X_train[i]) for i in range( X_train.shape[0])])

        self.X_train = np.array([np.pad( X_train[i], ((2,2),(2,2)), 'constant') for i in range( X_train.shape[0])])

        X_test = np.array([rgb2gray(X_test[i]) for i in range(X_test.shape[0])])

        self.X_test = np.array([np.pad(X_test[i], ((2,2),(2,2)), 'constant') for i in range(X_test.shape[0])])

    def load_data(self):
        
        return (self.X_train,self.y_train.reshape((self.y_train.shape[0]))), (self.X_test,self.y_test.reshape((self.y_test.shape[0])))

class Mnist():

    def __init__(self):

        self.name = 'Mnist'

        (self.X_train, self.y_train), (self.X_test,self.y_test)= mnist.load_data()
        
        
    def load_data(self):
        
        return (self. X_train,self.y_train.reshape((self.y_train.shape[0]))), (self.X_test,self.y_test.reshape((self.y_test.shape[0])))

class Fashion():
    
    def __init__(self):

        self.name = 'Fashion'

        (self.X_train, self.y_train), (self.X_test, self.y_test)= fashion_mnist.load_data()
        
        
    def load_data(self):
        
        return (self.X_train,self.y_train.reshape((self.y_train.shape[0]))), (self.X_test,self.y_test.reshape((self.y_test.shape[0])))


########Load dataset#######
def data_load(loaders):
    
    (X_train, y_train), (X_test, y_test) = loaders[0].load_data()

    if len(loaders)>1:

        for i in loaders[1:]:
        
             print('Loading '+i.name)

             (X_train_t, y_train_t), (X_test_t, y_test_t) = i.load_data()

             X_train = np.concatenate((X_train, X_train_t),axis=0)

             print( X_train.shape)

    return  X_train

########Clustering#########

def build_label(Cluster_labels,image_idxs):

    lenth = 0

    for i in image_idxs:

        lenth+=len(i)

    labels = [0]*lenth    

    for (i,j) in zip(Cluster_labels,image_idxs):

        for k in j:

            labels[k] = i

    return labels
        

def clustering(net,h_structure, X_train):

    all_images = []

    #Use a neural network as a feature extractor.
    #The extracted features will be transformed into 1-d array.
    #K-means will be applied on these 1-d arrays

    Net = net.eval()

    print("Start extracting features...\n")

    for i in tqdm(range(X_train.shape[0])):
        
        image = transforms.ToTensor()(X_train[i])

        image = image.to(torch.float32)
        
        image = image.unsqueeze(0)
        
        image = Net(image)
        
        image = image.reshape(-1, )
        
        all_images.append(image.detach().numpy())

        all_images_np = np.array(all_images)

    print("Start clustering...\n")

    labelIDs = []

    clts = []

    labels = []

    for i in range(len(h_structure)):

        print('Clustering' + str(i) + '-th layer\n' )

        if i == 0:

            clt = KMeans(n_clusters=h_structure[i])

            clt.fit(all_images_np)
    
            labelIDs.append(np.unique(clt.labels_))

            labels.append(clt.labels_)

            clts.append(clt)

        else:

            temp_images = []

            image_idxs = []

            for ID in labelIDs[i-1]:

                idxs = np.where(labels[i-1] == ID)[0]

                image_idxs.append(idxs)

                temp_images.append(np.concatenate(all_images_np[idxs]))
                
            clt = KMeans(n_clusters=h_structure[i])

            clt.fit(temp_images)

            temp_ids = clt.labels_

            labelIDs.append(np.unique(clt.labels_))

            labels.append(build_label(temp_ids,image_idxs))
            
    print("Clustering finished!\n")

    return labels

def build_datafile(net,h_structure, X_train,addr='./Data'):

    addr_store_original_train=addr+'/con/original/train/'

    if exists(addr_store_original_train+'label.txt'):
        
        os.remove(addr_store_original_train+'label.txt')    

    labels = clustering(net,h_structure,X_train,addr)

    print("Start building data file")

    for i in range(len(labels)):

        if i > 0:

            if len(labels[i])!=len(labels[i-1]):

                raise Exception("Error raised. Caused by different number of images in hierarchy.")

    with open(addr_store_original_train+'label.txt','a') as f1, open(addr_store_original_train+'class.txt','a') as f2:

            for i in range(len(X_train)):

                origin_img = Image.fromarray(X_train[i]).convert('L')

                dirName_o = addr+'/con/original/train/imgs/'+str(i)+'.png'

                origin_img.save(dirName_o)

                label_line = ' '

                for label in labels:

                    label_line = label_line + ' ' + str(label[i])

                f1.write(dirName_o+' '+label_line+'\n')

                label_line = ' '

            f2.write(str(labelID)+' '+str(1/class_num)+'\n')
            
    print("All done!")
        
if __name__ == '__main__':

    #Add dataset class to enable dataset
     
    dataset_loaders = [Clothing()]

    #Define feature extraction network
    net = ResNet18_f_extract()

    #Load Dataset
    X_train = data_load(dataset_loaders)

    #Start clustering and the clustered data will be stored at
    #./Data/con/original/train
    a = clustering(net, [5,3,2], X_train)

    print(a.shape())
    



    

    
        

        
        
        

        
