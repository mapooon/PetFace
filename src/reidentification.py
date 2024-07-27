import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from dataset import Reidentification
import pandas as pd
from backbones.resnet import r50

def main(args):

	device='cuda' if torch.cuda.is_available() else 'cpu'
	
	
	backbone = r50()
	state_dict=torch.load(args.weight, map_location='cpu')
	backbone.load_state_dict(state_dict['state_dict_backbone'])

	if args.mode=='arcface':
		state_dict_fc = state_dict['state_dict_softmax_fc']
		fc_weight=state_dict_fc['weight']
		fc_weight=fc_weight.to(device)
		norm_weight = F.normalize(fc_weight)
	elif args.mode=='center':
		centers=state_dict['state_dict_softmax_fc']['centers']
		centers=centers.to(device)
	elif args.mode=='softmax':
		state_dict_fc = state_dict['state_dict_softmax_fc']
		fc_weight=state_dict_fc['weight']
		fc_weight=fc_weight.to(device)
	else:
		raise NotImplementedError

	
	
	backbone=backbone.to(device)
	backbone.eval()
	
	dataset=Reidentification(args.input_csv,args.basedir)
	test_loader = DataLoader(
		dataset=dataset,
		batch_size=64,
		num_workers=4,
		pin_memory=True,
		drop_last=False,
		shuffle=False
	)
	sim_list=[]
	label_list=[]
	# pred_list=[]
	K=5
	pred_dict={f'top-{k+1}':[] for k in range(K)}
	for img,labels in tqdm(test_loader):
		with torch.no_grad():
			img=img.to(device)
			
			
			embeddings=backbone(img)

			#arcface
			if args.mode=='arcface':
				norm_embeddings=F.normalize(embeddings)
				logits = F.linear(norm_embeddings, norm_weight)
				_, indices = torch.topk(logits, K, dim=-1)
			elif args.mode=='center':
				x = embeddings
				batch_size = x.size(0)
				distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, centers.shape[0]) + \
						torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(centers.shape[0], batch_size).t()
				distmat.addmm_(1, -2, x, centers.t())

				_, indices = torch.topk(-distmat, K, dim=-1)

				

			elif args.mode=='softmax':
				
				logits = F.linear(embeddings, fc_weight)
				_, indices = torch.topk(logits, K, dim=-1)

			
			
			for k in range(K):
				pred_dict[f'top-{k+1}']+=indices[:,k].cpu().data.numpy().tolist()
			
			label_list+=labels.cpu().data.numpy().tolist()
			
	output_dict={'filename':dataset.img_list,'label':label_list}
	output_dict.update(pred_dict)
	pd.DataFrame(output_dict).to_csv(args.output,index=False)

		

			
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-w',dest="weight", type=str,required=True)
	parser.add_argument('-i',dest="input_csv", type=str,required=True)
	parser.add_argument('-o',dest="output", type=str,required=True)
	parser.add_argument('-b',dest="basedir", type=str,default='data/PetFace/images')
	parser.add_argument('-m',dest="mode", type=str,choices=['arcface','softmax','center'])
	main(parser.parse_args())
