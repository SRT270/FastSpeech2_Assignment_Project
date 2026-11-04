import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--logdir",required=True); p.add_argument("--output-dir",required=True)
    a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    ea=EventAccumulator(a.logdir); ea.Reload(); tags=ea.Tags().get("scalars",[])
    if not tags: raise SystemExit("No scalar tags found")
    print("Scalar tags:",tags)
    for filename,terms in [("training_loss.png",("loss","train")),("validation_loss.png",("loss","val","valid"))]:
        selected=[t for t in tags if any(x in t.lower() for x in terms)]
        if not selected: print("No tags match",filename); continue
        plt.figure(figsize=(9,5))
        for tag in selected:
            ev=ea.Scalars(tag)
            if ev: plt.plot([x.step for x in ev],[x.value for x in ev],label=tag)
        plt.xlabel("Step"); plt.ylabel("Loss"); plt.title(filename); plt.legend(); plt.grid(alpha=.25)
        plt.tight_layout(); plt.savefig(out/filename,dpi=160); plt.close()
if __name__=="__main__": main()
