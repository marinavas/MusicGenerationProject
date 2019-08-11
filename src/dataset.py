import imageio
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils import *
import scipy
import matplotlib as plt
#from sklearn.utils import shuffle
import random

project_path = ProjectPath("log")


class DataSet:
    """
    Abstract class that WGAN uses, needs to have a method which returns samples from the data
    """

    def next_batch_real(self, batch_size):
        """

        :param batch_size:
        :return: Tensor of real images in the shape [batch_size, height, width, channels]
        """
        raise NotImplementedError()

    def next_batch_fake(self, batch_size, z_size):
        return np.random.rand(batch_size, z_size)



class PianoRollData(DataSet):
    def __init__(self,img_size):
        self.img_size = img_size
        self.channels = 1
        #last_batch = 0
        #images_folder_path = os.path.join(project_path.base, "data", "Data64")
        self.x = np.load('data/data_x.npy')
        self.prev_x = np.load('data/data_prev.npy')

       # randomize = np.arange(len(self.x))
       # np.random.shuffle(randomize)
       # self.x = self.x[randomize]
       # self.prev_x = self.prev_x[randomize]        
        self.num_examples = len(self.x)
        print("Done")

    def next_batch_real(self, batch_size):
        locations = np.random.randint(0, self.num_examples, batch_size)
        #print(locations)
        imgs = np.array([self.x[i][:][:] for i in locations])
        img_prev = np.array([self.prev_x[i][:][:] for i in locations])
        #print(imgs[:,:,:,0].shape)
        t = np.reshape(imgs[:,:,:,0],(batch_size,16,128,1)) 
        tp = np.reshape(img_prev[:,:,:,0],(batch_size,16,128,1)) 
        
        return t, tp

    def get_image(self, path, resize_dim=None):
        img = PianoRollData.read_image(path)
        #print(img.shape)
        return img

    @staticmethod
    def read_image(file_path):
        # dividing with 256 because we need to get it in the [0, 1] range
        return imageio.imread(file_path).astype(np.float) / 256
      

    @staticmethod
    def center_crop(x, crop_h, crop_w=None):
        if crop_w is None:
            crop_w = crop_h
        h, w = x.shape[:2]
        j = round((h - crop_h) / 2)
        i = round((w - crop_w) / 2)
        return x[j:j + crop_h, i:i + crop_w]
