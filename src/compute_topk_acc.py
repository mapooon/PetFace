import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i',dest='input_csv',required=True)
parser.add_argument('--topk',type=int,default=1)
args = parser.parse_args()



df=pd.read_csv(args.input_csv)
K=args.topk

pred_list=np.array([df[f'top-{k+1}'].tolist() for k in range(K)]) 
labels=df['label'].tolist() 

topk_acc_list=[[] for _ in range(K)]
for i in range(len(labels)):
    label=labels[i]
    for k in range(K):
        topk_acc_list[k].append(label in pred_list[:k+1,i])
        
acc=sum(topk_acc_list[K-1])/len(labels)*100

print(acc)