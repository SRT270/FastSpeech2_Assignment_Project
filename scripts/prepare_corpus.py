import argparse, re
from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np

def safe_id(x):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(x))[:100] or "utt"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--language", choices=["English","Telugu"], required=True)
    p.add_argument("--split", choices=["train","val"], required=True)
    p.add_argument("--output-root", default="data/raw")
    a=p.parse_args()
    df=pd.read_csv(a.manifest)
    if not {"text","path"}.issubset(df.columns): raise SystemExit("CSV needs text,path columns")
    if df.empty or df[["text","path"]].isna().any().any(): raise SystemExit("Empty manifest/text/path")
    out=Path(a.output_root)/a.language/a.split/"speaker_001"
    out.mkdir(parents=True,exist_ok=True)
    seen=set(); secs=0
    for i,r in df.iterrows():
        src=Path(str(r.path))
        if not src.is_file(): raise FileNotFoundError(src)
        uid=safe_id(r["index"] if "index" in df.columns else i)
        if uid in seen: uid=f"{uid}_{i}"
        seen.add(uid)
        wav,sr=sf.read(src,dtype="float32")
        if wav.ndim==2: wav=wav.mean(axis=1)
        if len(wav)==0 or not np.isfinite(wav).all(): raise ValueError(f"Invalid audio: {src}")
        sf.write(out/f"{uid}.wav",wav,sr,subtype="PCM_16")
        (out/f"{uid}.lab").write_text(str(r.text).strip(),encoding="utf-8")
        secs += len(wav)/sr
    print(f"{len(df)} WAV/LAB pairs written to {out}; {secs/3600:.3f} hours")
    print("Forced-alignment TextGrids are still required.")
if __name__=="__main__": main()
