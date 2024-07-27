import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from backbones.iresnet import iresnet50
from backbones.swin import swinb
from backbones.resnet import r50
import numpy as np
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from dataset import Verification
import pandas as pd
from backbones import get_model
import os

def main(args):

	device='cuda' if torch.cuda.is_available() else 'cpu'

	if args.network is not None:
		arcface=get_model(args.network, dropout=0.0, fp16=False, num_features=512)
	else:
		arcface = r50()
	arcface.load_state_dict(torch.load(args.weight, map_location='cpu')['state_dict_backbone'])
	arcface=arcface.to(device)
	arcface.eval()
	
	dataset=Verification(args.input_csv,args.basedir)
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
	for img1,img2,labels in tqdm(test_loader):
		with torch.no_grad():
			img1=img1.to(device)
			img2=img2.to(device)
			
			vec1=F.normalize(arcface(img1))
			vec2=F.normalize(arcface(img2))

			sim=nn.CosineSimilarity()(vec1,vec2).cpu().data.numpy().tolist()
			sim_list+=sim
			label_list+=labels.cpu().data.numpy().tolist()
			
	os.makedirs(os.path.dirname(args.output),exist_ok=True)
	pd.DataFrame({'filename1':dataset.img1_list,'filename2':dataset.img2_list,'sim':sim_list,'label':label_list}).to_csv(args.output,index=False)

		

			
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-w',dest="weight", type=str,required=True)
	parser.add_argument('-i',dest="input_csv", type=str,required=True)
	parser.add_argument('-o',dest="output", type=str,required=True)
	parser.add_argument('-b',dest="basedir", type=str,default='data/PetFace/images')
	parser.add_argument("--network", type=str, default=None)
	main(parser.parse_args())
