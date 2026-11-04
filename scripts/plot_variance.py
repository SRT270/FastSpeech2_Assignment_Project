import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--title",default="Variance adaptor"); a=p.parse_args()
    src=Path(a.input_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    for name,label in [("duration","Predicted duration (frames)"),("pitch","Predicted pitch / F0"),("energy","Predicted energy")]:
        f=src/f"{name}.npy"
        if not f.exists(): print("Missing",f); continue
        v=np.asarray(np.load(f)).squeeze()
        if not v.size: print("Empty",f); continue
        plt.figure(figsize=(10,4)); plt.plot(np.arange(v.size),v)
        plt.xlabel("Token/frame index"); plt.ylabel(label); plt.title(f"{a.title}: {name}")
        plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/f"{name}.png",dpi=160); plt.close()
if __name__=="__main__": main()
