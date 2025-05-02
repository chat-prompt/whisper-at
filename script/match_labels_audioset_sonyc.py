import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

def load_sonyc_labels(csv_path: str) -> list:
    df = pd.read_csv(csv_path)
    presence_cols = [col for col in df.columns if col.endswith("_presence")]
    cleaned = sorted(set(['_'.join(col.replace('_presence', '').split('_')[1:]) for col in presence_cols]))
    return cleaned

def load_audioset_ontology(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        ontology = json.load(f)
    audioset_data = {}
    for item in ontology:
        if item["restrictions"] == ["abstract"]:
            continue
        label = item["name"]
        description = item.get("description", "")
        audioset_data[label] = description
    return audioset_data

def match_labels(sonyc_labels: list, audioset_data: dict, threshold=0.6):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight and fast
    mapping = {}

    audioset_labels = list(audioset_data.keys())
    audioset_descs = [f"{k}: {v}" for k, v in audioset_data.items()]
    audioset_embs = model.encode(audioset_descs, convert_to_tensor=True)

    for s_label in sonyc_labels:
        s_query = s_label.replace("-", " ").replace("_", " ")
        s_emb = model.encode(s_query, convert_to_tensor=True)

        scores = util.cos_sim(s_emb, audioset_embs)[0]
        best_score, best_idx = float(scores.max()), int(scores.argmax())

        if best_score >= threshold:
            mapping[s_label] = audioset_labels[best_idx]
        else:
            mapping[s_label] = None

    return mapping

def save_mapping(mapping, out_path="semantic_sonyc_to_audioset_mapping.json"):
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"✅ 매핑 결과 저장 완료: {out_path}")

# 실행 예시
if __name__ == "__main__":
    sonyc_csv = "script/annotations.csv"
    ontology_json = "script/ontology.json"
    output_file = "script/semantic_sonyc_to_audioset_mapping.json"

    sonyc_labels = load_sonyc_labels(sonyc_csv)
    audioset_ontology = load_audioset_ontology(ontology_json)
    print(f"label_len: {len(sonyc_labels)}, audioset_ontology_len: {len(audioset_ontology)}")
    mapping = match_labels(sonyc_labels, audioset_ontology, threshold=0.6)
    save_mapping(mapping, output_file)
