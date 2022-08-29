import pickle
import cv2
import numpy as np
from tqdm.notebook import trange, tqdm

# file_name -> path + name of the file
def load_images(file_name):
    # get file content
    with open(file_name, 'rb') as f:
        info = pickle.load(f)

    img_data = info['image_data']
    class_dict = info['class_dict']

    # create arrays to store x and y of images
    images = [] # x
    labels = [] # y
  
    # loop over all images and store them
    loading_msg = 'Reading images from %s' % file_name

    # loop over all classes
    for item in tqdm(class_dict.items(), desc = loading_msg):
        # loop over all examples from the class
        for example_num in item[1]:
            # convert image to RGB color channels
            RGB_img = cv2.cvtColor(img_data[example_num], cv2.COLOR_BGR2RGB)

            # store image and corresponding label
            images.append(RGB_img)
            labels.append(item[0])
  
    # return set of images
    return np.array(images), np.array(labels)

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

# img_set_x -> images
# img_set_y -> labels
# num_way -> number of classes for episode
# num_shot -> number of examples per class
# num_query -> number of query examples per class 
def extract_episode(img_set_x, img_set_y, num_way, num_shot, num_query):
    # get a list of all unique labels (no repetition)
    unique_labels = np.unique(img_set_y)

    # select num_way classes randomly without replacement
    chosen_labels = np.random.choice(unique_labels, num_way, replace = False)
    # number of examples per selected class (label)
    examples_per_label = num_shot + num_query

    # list to store the episode
    episode = []

    # iterate over all selected labels 
    for label_l in chosen_labels:
        # get all images with a certain label l
        images_with_label_l = img_set_x[img_set_y == label_l]

        # suffle images with label l
        shuffled_images = np.random.permutation(images_with_label_l)

        # chose examples_per_label images with label l
        chosen_images = shuffled_images[:examples_per_label]

        # add the chosen images to the episode
        episode.append(chosen_images)

    # turn python list into a numpy array
    episode = np.array(episode)

    # convert numpy array to tensor of floats
    episode = torch.from_numpy(episode).float()

    # reshape tensor (required)
    episode = episode.permute(0,1,4,2,3)

    # get the shape of the images
    img_dim = episode.shape[2:]
  
    # build a dict with info about the generated episode
    episode_dict = {
        'images': episode, 'num_way': num_way, 'num_shot': num_shot, 
        'num_query': num_query, 'img_dim': img_dim}

    return episode_dict

# episode_dict -> dict with info about the chosen episode
def display_episode_images(episode_dict):
    # number of examples per class 
    examples_per_label = episode_dict['num_shot'] + episode_dict['num_query']

    # total number of images
    num_images = episode_dict['num_way'] * examples_per_label

    # select the images
    images = episode_dict['images'].view(num_images, *episode_dict['img_dim'])

    # create a grid with all the images
    grid_img = torchvision.utils.make_grid(images, nrow = examples_per_label)

    # reshape the tensor and convert to numpy array of integers 
    grid_img = grid_img.permute(1, 2, 0).numpy().astype(np.uint8)

    # chage image from BGR to RGB
    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)

    # set a bigger size
    plt.figure(figsize = (80, 8))

    # remove the axis
    plt.axis('off')

    # plot the grid image
    plt.imshow(grid_img)

import torch.nn as nn
import torch.optim as optim

import numpy as np
from math import fsum

from tqdm.notebook import trange
from os import path, mkdir

# Options: "omniglot" or "mini_imagenet"
dirc = '/home/atik/Documents/Prototypical Networks/'
dataset = dirc+"mini_imagenet"

data_dir = path.join(dataset, 'data')

train_x, train_y = load_images(path.join(data_dir,  'train.pkl'))
valid_x, valid_y = load_images(path.join(data_dir,  'valid.pkl'))
test_x, test_y = load_images(path.join(data_dir,  'test.pkl'))

def label_generate(y):
    unique = np.unique(y)
    temp = []
    for i in range(len(unique)):
        m = np.where(y == unique[i])
        n = np.full(len(m[0]), i)
        temp.append(n)
    return np.array(temp).flatten()  

train_y = label_generate(train_y)
valid_y = label_generate(valid_y)
test_y = label_generate(test_y)

# np.save(dirc+'valid_x', valid_x)
# np.save(dirc+'valid_y', valid_y)

episode_dict = extract_episode(train_x, train_y, num_way = 5, num_shot = 5, num_query = 5)
display_episode_images(episode_dict)

""""""
import torch.nn.functional as F
from torch.autograd import Variable

def euclidean_dist(x, y):
    elements_in_x = x.size(0)
    elements_in_y = y.size(0)

    dimension_elements = x.size(1)

    assert dimension_elements == y.size(1)

    x = x.unsqueeze(1).expand(elements_in_x , elements_in_y, dimension_elements)
    y = y.unsqueeze(0).expand(elements_in_x , elements_in_y, dimension_elements)

    distances = torch.pow(x - y, 2).sum(2)

    return distances

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super(ProtoNet, self).__init__()
        self.encoder = encoder.cuda()

    def set_forward_loss(self, episode_dict):
        # extract all images 
        images = episode_dict['images'].cuda()
        
        # get episode setup
        num_way = episode_dict['num_way'] # way
        num_shot = episode_dict['num_shot'] # shot
        num_query = episode_dict['num_query'] # number of query images
        
        # from each class, extract num_shot support images
        x_support = images[:, :num_shot] # lines are classes and columns are images
        
        # from each class, extract the remaining images as query images
        x_query = images[:, num_shot:] # lines are classes and columns are images
        
        # create indices from 0 to num_way-1 for classification
        target_inds = torch.arange(0, num_way).view(num_way, 1, 1)
        
        # replicate all indices num_query times (for each query image)
        target_inds = target_inds.expand(num_way, num_query, 1).long()
        
        # convert indices from Tensor to Variable
        target_inds = Variable(target_inds, requires_grad = False).cuda()
        
        # transform x_support into a array in which all images are contiguous
        x_support = x_support.contiguous().view(
            num_way * num_shot, *x_support.size()[2:]) # no more lines and columns
                
        # transform x_query into a array in which all images are contiguous
        x_query = x_query.contiguous().view(
            num_way * num_query, *x_query.size()[2:]) # no more lines and columns

        # join all images into a single contiguous array 
        x = torch.cat([x_support, x_query], 0)
        
        # encode all images
        z = self.encoder.forward(x) # embeddings
        
        # compute class prototypes
        z_dim = z.size(-1)
        z_proto = z[:(num_way * num_shot)].view(num_way, num_shot, z_dim).mean(1)
        
        # get the query embeddings
        z_query = z[(num_way * num_shot):]

        # compute distance between query embeddings and class prototypes
        dists = euclidean_dist(z_query, z_proto)
        
        # compute the log probabilities
        log_p_y = F.log_softmax(-dists, dim = 1).view(num_way, num_query, -1)
        
        # compute the loss
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        
        # get the predicted labels for each query
        _, y_hat = log_p_y.max(2) # lines are classes and columns are query embeddings
        
        # compute the accuracy
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        
        # return output: loss, acc and predicted value
        return loss_val, {
            'loss': loss_val.item(), 'acc': acc_val.item(), 'y_hat': y_hat}

# function to load the model structure
def load_protonet(x_dim, hid_dim, z_dim):
    # define a convolutional block
    def conv_block(layer_input, layer_output): 
        conv = nn.Sequential(
            nn.Conv2d(layer_input, layer_output, 3, padding=1),
            nn.BatchNorm2d(layer_output), nn.ReLU(), 
            nn.MaxPool2d(2))

        return conv
  
    # create the encoder to the embeddings for the images
    # the encoder is made of four conv blocks 
    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim), conv_block(hid_dim, hid_dim), 
        conv_block(hid_dim, hid_dim), conv_block(hid_dim, z_dim), Flatten())
  
    return ProtoNet(encoder)







