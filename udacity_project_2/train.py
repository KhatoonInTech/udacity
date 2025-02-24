import argparse
import time
import os, random
import torch
from utils import data_transform, data_loader, set_up_model, train_model, save_checkpoints

def main():
    
    print('Training Started...\n')
    
    # parsing arguments
    parser = argparse.ArgumentParser(description='Model training program')

    parser.add_argument('--data_dir', dest='data_dir', required=True, help='Path of data directory.')
    parser.add_argument('--save_dir', dest='save_dir', help='Set directory to save checkpoints.')
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg13', 'vgg16', 'vgg19'], help='Select the architecture.')
    parser.add_argument('--dropout', dest = "dropout", type=float, default = 0.3, help = "set the dropout probability")
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001', help='Set the learning rate.')
    parser.add_argument('--hidden_units', dest='hidden_units', default='4096', help='Set hidden units.')
    parser.add_argument('--epochs', dest='epochs', default='5', help='Set epochs.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU.')
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    dropout=args.dropout
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu
    
    # check GPU arg
    if gpu==True:
        device = 'cuda'
        print('Using CUDA\n')
    else:
        device = 'cpu'
        print('Using CPU\n')
        
    # print args
    print(f'Architecture: {arch}')
    print(f'Learning Rate: {learning_rate}')
    print(f'Hidden Layer: {hidden_units}')
    print(f'Epochs: {epochs}')
    print(f'Data Directory: {data_dir}')
    print(f'Save Directory: {save_dir}\n')
        
    # check checkpoint path availability
    if save_dir==None:
        save_dir = 'saved_models'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    cat_output  = len(cat_to_name)
    
    train_data, valid_data, test_data = data_transform(data_dir)
    
    trainloader,validloader,testloader = data_loader(train_data, valid_data, test_data)
    
    model, criterion, optimizer = set_up_model(arch=arch, dropout=dropout, lr=learning_rate, hidden_units=hidden_units, len=cat_output)
     
    train_model(epochs, model, optimizer, criterion,  trainloader, validloader, device)
    
    save_checkpoints(model, cat_output, train_data, path=save_dir)    

if __name__ == '__main__':
    main()