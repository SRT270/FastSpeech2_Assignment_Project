from pathlib import Path
import yaml, json

def test_configs():
    for lang in ("English","Telugu"):
        c=yaml.safe_load((Path(lang)/"config.yaml").read_text(encoding="utf-8"))
        assert c["language"]==lang
        assert c["model"]["n_mel_channels"]==80

def test_test_sentence_counts():
    d=json.loads(Path("data/test_sentences.json").read_text(encoding="utf-8"))
    assert len(d["English"])==5
    assert len(d["Telugu"])==5
