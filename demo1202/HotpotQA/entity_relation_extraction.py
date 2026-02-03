import json
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
from tqdm import tqdm

def safe_parse_json(text: str) -> Dict:
    """
    Try to extract JSON from LLM output.
    Robust against minor formatting issues.
    """
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"entities": [], "relations": []}


def normalize_entity(entity: str) -> str:
    return entity.strip()


def normalize_relation(rel: Dict) -> Dict:
    return {
        "head": normalize_entity(rel.get("head", "")),
        "relation": rel.get("relation", "").strip(),
        "tail": normalize_entity(rel.get("tail", ""))
    }

class LLMExtractor:
    def __init__(self, model_name: str, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32
        )

    def extract(self, text: str, max_new_tokens: int = 256) -> str:
        prompt = self._build_prompt(text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @staticmethod
    def _build_prompt(text: str) -> str:
        prompt = f"""You are an information extraction system.
        From the text below, extract:
        1. Named entities
        2. Semantic relations between entities

        Return a JSON object with this format:
        {{
            "entities": ["entity1", "entity2"],
            "relations": [
                {{
                "head": "entity1",
                "relation": "relation phrase",
                "tail": "entity2"
                }}
            ]
        }}

        Text:
        {text}
        """

        return prompt


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

INPUT_CHUNKS = Path("./processed/chunks.jsonl")
OUTPUT_KG = Path("./processed/kg.jsonl")

def main():
    extractor = LLMExtractor(MODEL_NAME)

    OUTPUT_KG.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_CHUNKS) as fin, open(OUTPUT_KG, "w") as fout:
        for line in tqdm(fin, desc="Extracting KG"):
            chunk = json.loads(line)

            text = chunk["text"]
            chunk_id = f"{chunk['doc_title']}::{chunk['chunk_id']}"

            raw_output = extractor.extract(text)
            parsed = safe_parse_json(raw_output)

            # Entities
            entities = list(set(normalize_entity(e) for e in parsed.get("entities", [])))


            # Relations
            relations = [
                normalize_relation(r)
                for r in parsed.get("relations", [])
                if r.get("head") and r.get("tail")
            ]

            fout.write(json.dumps({
                "chunk_id": chunk_id,
                "entities": entities,
                "relations": relations
            }) + "\n")


if __name__ == "__main__":
    main()
