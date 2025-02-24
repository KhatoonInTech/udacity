import argparse
import time
import os, random
import json
import torch
from utils import load_checkpoint,predict, show_pred

def main():
    
    print('Training Started...\n')
    
    # parsing arguments
    parser = argparse.ArgumentParser(description='Prediction program')

    parser.add_argument('input_img', default="flowers/test/1/image_06752.jpg", help = "path of train dataset", metavar="FILE")
#     parser.add_argument('--gpu', action="store_true", default="gpu", help = "Use GPU or CPU to train model")
    parser.add_argument('checkpoint_path', action="store", default="checkpoint.pth", help = "accessing saved checkpoint")
    parser.add_argument('--top_k', default=5, dest="top_k", type=int)
    parser.add_argument('--cat_names', dest="cat_names", default='cat_to_name.json')
    
    args = parser.parse_args()

    input_img = args.input_img
    checkpoint_path = args.checkpoint_path
    topk = args.top_k
    cat_names = args.cat_names
#     gpu = args.gpu
    
    # check GPU arg
#     if gpu==True:
#         device = 'cuda'
#         print('Using CUDA\n')
#     else:
#         device = 'cpu'
#         print('Using CPU\n')
    
    with open('cat_to_name.json', 'r') as f:
           cat_to_name = json.load(f)    
    
    # print args
    print(f'Input Image Path: {input_img}')
    print(f'Checkpoint Path: {checkpoint_path}')
    print(f'topk: {topk}')
    print(f'cat_names: {cat_names}')
#     print(f'gpu: {gpu}')
        
            
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    cat_output  = len(cat_to_name)
    
 
    model=load_checkpoint(checkpoint_path)
    prob, classes = predict(input_img, model, topk)
    show_pred(prob, classes, cat_to_name)
    
if __name__ == '__main__':
    main()