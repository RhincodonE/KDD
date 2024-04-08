from tensorflow.keras.datasets import cifar10,cifar100,fashion_mnist,mnist
from emnist import extract_train_samples,extract_test_samples
import numpy as np
import PIL
from PIL import Image
import csv
import os
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

        X_train = np.array([skimage.measure.block_reduce(X_train[i],(10,10),np.mean) for i in range(X_train.shape[0])])

        print(X_train.shape)
        self.X = np.array([np.pad(X_train[i], ((0,1),(0,1)), 'constant') for i in range(X_train.shape[0])])
        print(self.X.shape)
        self.y = np.array(label)
        print(self.y.shape)

    def load_data(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=3)
 
        return (X_train,y_train.reshape((y_train.shape[0]))), (X_test,y_test.reshape((y_test.shape[0])))

class Clothing():

    def __init__(self):

        self.name = 'Clothing'

        image_size = (36, 36)
        
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

        print (X[0])

        self.X =  np.array([rgb2gray(X[i]) for i in range(X.shape[0])])

        print(self.X[0])
                        
        y = np.vstack(y)

        self.y = np.argmax(y,axis=1)


    def load_data(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=3)
 
        return (X_train,y_train.reshape((y_train.shape[0]))), (X_test,y_test.reshape((y_test.shape[0])))

class Emnist():

    def __init__(self):

        self.name = 'Emnist'

        X_train, y_train = extract_train_samples('letters')

        X_test, y_test = extract_test_samples('letters')

        y_train = y_train.reshape((y_train.shape[0]))

        y_test = y_test.reshape((y_test.shape[0]))

        X_train = np.array([np.pad(X_train[i], ((4,4),(4,4)), 'constant') for i in range(X_train.shape[0])])

        X_test = np.array([np.pad(X_test[i], ((4,4),(4,4)), 'constant') for i in range(X_test.shape[0])])

        elements = [10,1,2,3,4,5,6,7,8,9]

        ind_train = search_index(elements,y_train)
        
        ind_test = search_index(elements,y_test)
        
        self.X_train = X_train[ind_train]

        self.y_train = y_train[ind_train]

        self.X_test = X_test[ind_test]

        self.y_test = y_test[ind_test]

        self.y_train[self.y_train == 10] = 0
        
        self.y_test[self.y_test == 10] = 0
        
    def load_data(self):
        
        return (self.X_train,self.y_train.reshape((self.y_train.shape[0]))), (self.X_test,self.y_test.reshape((self.y_test.shape[0])))

class Cifar100():

    def __init__(self):

        self.name = 'Cifar100'

        (X_train, y_train), (X_test, y_test)= cifar100.load_data()

        y_train = y_train.reshape((y_train.shape[0]))

        y_test = y_test.reshape((y_test.shape[0]))

        X_train = np.array([rgb2gray(X_train[i]) for i in range(X_train.shape[0])])

        X_train = np.array([np.pad(X_train[i], ((2,2),(2,2)), 'constant') for i in range(X_train.shape[0])])

        X_test = np.array([rgb2gray(X_test[i]) for i in range(X_test.shape[0])])

        X_test = np.array([np.pad(X_test[i], ((2,2),(2,2)), 'constant') for i in range(X_test.shape[0])])

        #elements = random.randint(0,99,(10))

        elements = [0,1,2,3,4,5,6,7,8,9]

        ind_train = search_index(elements,y_train)
        
        ind_test = search_index(elements,y_test)

        self.X_train = X_train[ind_train]

        self.y_train = y_train[ind_train]

        self.X_test = X_test[ind_test]

        self.y_test = y_test[ind_test]

    def load_data(self):
        
        return (self.X_train,self.y_train.reshape((self.y_train.shape[0]))), (self.X_test,self.y_test.reshape((self.y_test.shape[0])))

class Cifar10():

    def __init__(self):

        self.name = 'Cifar10'

        (X_train, self.y_train), (X_test, self.y_test)= cifar10.load_data()

        self.y_train = self.y_train.reshape((self.y_train.shape[0]))

        self.y_test = self.y_test.reshape((self.y_test.shape[0]))

        X_train = np.array([rgb2gray(X_train[i]) for i in range(X_train.shape[0])])

        self.X_train = np.array([np.pad(X_train[i], ((2,2),(2,2)), 'constant') for i in range(X_train.shape[0])])

        X_test = np.array([rgb2gray(X_test[i]) for i in range(X_test.shape[0])])

        self.X_test = np.array([np.pad(X_test[i], ((2,2),(2,2)), 'constant') for i in range(X_test.shape[0])])

    def load_data(self):
        
        return (self.X_train,self.y_train.reshape((self.y_train.shape[0]))), (self.X_test,self.y_test.reshape((self.y_test.shape[0])))

class Mnist():

    def __init__(self):

        self.name = 'Mnist'

        (X_train, self.y_train), (X_test, self.y_test)= mnist.load_data()
        
        self.X_train = np.array([np.pad(X_train[i], ((4,4),(4,4)), 'constant') for i in range(X_train.shape[0])])

        self.X_test = np.array([np.pad(X_test[i], ((4,4),(4,4)), 'constant') for i in range(X_test.shape[0])])

        
    def load_data(self):
        
        return (self.X_train,self.y_train.reshape((self.y_train.shape[0]))), (self.X_test,self.y_test.reshape((self.y_test.shape[0])))

class Fashion():
    
    def __init__(self):

        self.name = 'Fashion'

        (X_train, self.y_train), (X_test, self.y_test)= fashion_mnist.load_data()
        
        self.X_train = np.array([np.pad(X_train[i], ((4,4),(4,4)), 'constant') for i in range(X_train.shape[0])])

        self.X_test = np.array([np.pad(X_test[i], ((4,4),(4,4)), 'constant') for i in range(X_test.shape[0])])

        
    def load_data(self):
        
        return (self.X_train,self.y_train.reshape((self.y_train.shape[0]))), (self.X_test,self.y_test.reshape((self.y_test.shape[0])))

        
def data_load(addr,loader):
    
    print('Loading '+loader.name)
    
    (X_train, y_train), (X_test, y_test) = loader.load_data()

    print(X_train.shape)

    print(y_train.shape)

    print(X_test.shape)

    print(y_test.shape)

    count=0
    
    second_level_dir = addr+'/'+loader.name

    if not exists(second_level_dir):

        os.mkdir(second_level_dir)
    
    addr_store_train = second_level_dir+'/train/'

    if not exists(addr_store_train):

        os.mkdir(addr_store_train)

    addr_store_test = second_level_dir+'/test/'

    if not exists(addr_store_test):

        os.mkdir(addr_store_test)

    imgs_dir_train = addr_store_train+'/imgs/'
    
    if not exists(imgs_dir_train):

        os.mkdir(imgs_dir_train)

    imgs_dir_test = addr_store_test+'/imgs/'
    
    if not exists(imgs_dir_test):

        os.mkdir(imgs_dir_test)
        
    if exists(addr_store_train+'label.txt'):
        
        os.remove(addr_store_train+'label.txt')
        
    if exists(addr_store_test+'label.txt'):
        
        os.remove(addr_store_test+'label.txt')

    with open(addr_store_train+'label.txt','a') as f1:
        
        for i in X_train:

            img = Image.fromarray(i).convert('L')

            dirName = imgs_dir_train+str(count)+'.png'

            img.save(dirName)

            f1.write(dirName+' '+str(y_train[count])+'\n')

            count+=1
        
    count=0
    
    with open(addr_store_test+'label.txt','a') as f3:
        
        for i in X_test:

            img = Image.fromarray(i).convert('L')

            dirName = imgs_dir_test+str(count)+'.png'

            img.save(dirName)

            f3.write(dirName+' '+str(y_test[count])+'\n')
            
            count+=1

    print(loader.name+' loading finished\n')

if __name__ == '__main__':
    
    dataset_loaders = [Fashion()]

    for loader in dataset_loaders:
        
        data_load('./Data_train',loader)



    

    
        

        
        
        

        
