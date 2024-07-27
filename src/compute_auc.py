import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve


parser = argparse.ArgumentParser()
parser.add_argument('-i',dest='input_csv',required=True)
args = parser.parse_args()

df=pd.read_csv(args.input_csv)
sims=np.array(df['sim'].tolist())
sims=((sims>0)*sims).tolist()
labs=df['label'].tolist()

auc = roc_auc_score(labs, sims)*100
    
print(auc)
