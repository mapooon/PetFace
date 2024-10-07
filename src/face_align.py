import numpy as np
import cv2
import skimage
import skimage.transform
import argparse


def align(img,lmk,src,size):
	simtf=skimage.transform.SimilarityTransform()
	simtf.estimate(lmk,src)
	M=simtf.params.copy()
	img_aligned = cv2.warpPerspective(img,M,(size,size),flags=cv2.INTER_AREA)
	return img_aligned

def main(args):
	img = cv2.imread(args.img)
	h,w=img.shape[:2]
	src = np.load(args.src).reshape((5,2))
	tgt = np.load(args.tgt)[0].reshape((5,2))
	img_aligned = align(img,tgt,src,224)
	cv2.imwrite(args.out,img_aligned)


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--src", type=str, help="source landmarks path", required=True)
	parser.add_argument("--tgt", type=str, help="your image's landmarks path", required=True)
	parser.add_argument("--img", type=str, help="your image path", required=True)
	parser.add_argument("--out", type=str, help="output path", required=True)
	main(parser.parse_args())
	