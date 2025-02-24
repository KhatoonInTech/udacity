import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os, random
from PIL import Image
import torch
from torch import nn, optim, from_numpy
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler

def data_transform(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # transforming train set for more training examples
    train_transforms = transforms.Compose([
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                          ])

    # transforming valid set
    valid_transforms = transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                         ])

    # transforming test set (same as valid set)
    test_transforms = transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                         ])
        
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    return train_data, valid_data, test_data


def data_loader(train_data, valid_data, test_data):
  
    
    #  Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloader,validloader,testloader

def set_up_model(arch="vgg16", dropout="0.3", lr=0.001, hidden_units=512, len=102):
    '''
    Arguments: The architecture for the network(resnet50,densenet121,vgg16), the hyperparameters for the network (dropout, learning rate) and use gpu or not
    Returns: Set up model with NLLLoss() and Adam optimizer
    '''

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Sorry {} model is not available. Please select vgg16 or densenet121".format(arch))

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(in_features[arch], hidden_units),
                              nn.ReLU(),
                              nn.Dropout(dropout),
                              nn.Linear(hidden_units, len),
                              nn.LogSoftmax(dim=1)
                             )
    
    
    model.classifier = classifier
        
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    criterion = nn.NLLLoss()

    return model, criterion, optimizer

# Pre-fetch data to device if possible
def prefetch_data(dataloader, device):
    try:
        return [(inputs.to(device), labels.to(device)) 
                for inputs, labels in dataloader]
    except RuntimeError:  # Handle out of memory
        return None
    
def validate_and_print(model, validation_data, criterion, running_loss, steps, device, start_time):
    model.eval()
    valid_loss = 0
    accuracy = 0
    print_every=10
    with torch.no_grad():
        for inputs, labels in validation_data:
            if isinstance(validation_data, list):
                val_inputs, val_labels = inputs, labels
            else:
                val_inputs = inputs.to(device)
                val_labels = labels.to(device)
                
            logps = model(val_inputs)
            batch_loss = criterion(logps, val_labels)
            valid_loss += batch_loss.item()
            
            # Efficient accuracy calculation
            predictions = logps.argmax(dim=1)
            accuracy += (predictions == val_labels).float().mean().item()
    
    # Calculate metrics
    avg_train_loss = running_loss / print_every
    avg_valid_loss = valid_loss / len(validation_data)
    avg_accuracy = accuracy / len(validation_data)
    time_taken = time.time() - start_time
    
    print(f"Step {steps}.. "
          f"Train loss: {avg_train_loss:.3f}.. "
          f"Validation loss: {avg_valid_loss:.3f}.. "
          f"Validation accuracy: {avg_accuracy:.3f}.. "
          f"Time: {time_taken:.3f}s")


#training data
def train_epoch(model, trainloader, validloader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    steps = 0
    
    # Try to prefetch validation data
    valid_data = prefetch_data(validloader, device) if device.type == 'cuda' else None
    
    for inputs, labels in trainloader:
        steps += 1
        start = time.time()
        
        # Transfer to device efficiently
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass and loss calculation
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        logps = model(inputs)
        loss = criterion(logps, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            validate_and_print(model, valid_data or validloader, criterion, 
                             running_loss, steps, device, start)
            running_loss = 0
            model.train()

#MODEL TRAIING
def train_model(epochs,model, optimizer, criterion,  trainloader, validloader, device, print_every):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_epoch(model, trainloader, validloader, optimizer, criterion, device, print_every)

# Save the CheckPoints
def save_checkpoints(model, cat_output, train_data, path='checkpoint.pth'):
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': 2048,
                  'output_size': cat_output,
                  'arch': model,
                  'fc' : model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, path)


#load the model with checkpoints
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    model = checkpoint['arch']
    model.fc = checkpoint['fc']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
        
    img_pil = Image.open(image)

    # define transforms

    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                          std=[0.229, 0.224, 0.225])
                                    ])
    
    img_tensor = preprocess(img_pil)
    np_image = np.array(img_tensor)
    
    return np_image



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # predict the class from an image file
    image = torch.from_numpy(process_image(image_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, image = model.to(device), image.to(device, dtype=torch.float)
    model.eval()
    
    output = model(image.unsqueeze(0)) 
    ps = torch.exp(output)
    
    # getting the topk (=5) probabilites and indexes
    prob = torch.topk(ps, topk)[0].tolist()[0] # probabilities
    index = torch.topk(ps, topk)[1].tolist()[0] # index
    
    idx = []
    for i in range(len(model.class_to_idx.items())):
        idx.append(list(model.class_to_idx.items())[i][0])
        
    classes = []
    for i in range(topk):
        classes.append(idx[index[i]])
    
    return prob, classes

def show_pred(prob, classes, cat_to_name):
    print(prob)
    print([cat_to_name[i] for i in classes])
    print(f"The flower is {cat_to_name[classes[0]]}")