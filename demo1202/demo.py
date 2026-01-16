from lightrag import LightRAG, QueryParam
import json

# ---------------------------------------------------------
# Load KG triples from MetaQA-style data
# ---------------------------------------------------------
def load_kg(path):
    triples = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                s, p, o = parts[0], parts[1], parts[2]
                triples.append((s, p, o))
    return triples

# ---------------------------------------------------------
# Load QA pairs
# ---------------------------------------------------------
def load_qa(path):
    qa_pairs = []
    with open(path, "r") as f:
        for line in f:
            if "\t" in line:
                q, a = line.strip().split("\t")
                qa_pairs.append((q, a))
    return qa_pairs

# ---------------------------------------------------------
# Initialize LightRAG
# ---------------------------------------------------------
# You may replace llm_model with your local LLM such as:
# llm_model="ollama:deepseek-r1:latest"
# Initialize LightRAG with Hugging Face model
from lightrag import LightRAG, EmbeddingFunc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch


# ---------------------------------------------------------
# Load HuggingFace LLM (Qwen2-2B)
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

llm_name = "Qwen/Qwen2-2B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Custom text generation function
def hf_model_complete(prompt: str) -> str:
    """Run LLM generation using HuggingFace Qwen2-2B."""
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3
    )
    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------------------------------------------------
# Load embedding model (BGE-M3 or MiniLM)
# ---------------------------------------------------------
embed_name = "BAAI/bge-m3"
embed_tokenizer = AutoTokenizer.from_pretrained(embed_name)
embed_model = AutoModel.from_pretrained(embed_name).to(device)


def hf_embed(texts):
    """Return embeddings for a list of texts using HF embedding model."""
    encoded_input = embed_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        model_output = embed_model(**encoded_input)

    # Mean pooling
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()


# ---------------------------------------------------------
# Initialize LightRAG with custom HF models
# ---------------------------------------------------------
rag = LightRAG(
    working_dir="./kgqa_qwen2_hf",
    llm_model_func=hf_model_complete,  # HuggingFace LLM
    llm_model_name=llm_name,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,   # bge-m3 embedding dimension
        func=hf_embed          # your embedding function
    )
)

print("LightRAG with Qwen2-2B + HF embeddings initialized!")


# ---------------------------------------------------------
# Insert KG into LightRAG
# ---------------------------------------------------------
kg = load_kg("kg.txt")
print(f"Loading KG triples: {len(kg)}")

for s, p, o in kg:
    # Insert structured triple into LightRAG KG storage
    rag.insert_triplet(s, p, o)

print("KG inserted successfully.")

# ---------------------------------------------------------
# Load QA data
# ---------------------------------------------------------
qa_data = load_qa("qa.txt")
print(f"Loaded QA pairs: {len(qa_data)}")

# ---------------------------------------------------------
# Evaluate KGQA using LightRAG
# ---------------------------------------------------------
correct = 0
results = []

for q, gold in qa_data:
    # Use global mode (KG-based retrieval)
    ans = rag.query(q, mode="global")

    # Normalize strings
    ans_clean = ans.lower().replace(" ", "").strip()
    gold_clean = gold.lower().replace(" ", "").strip()

    is_correct = gold_clean in ans_clean
    correct += int(is_correct)

    results.append({
        "question": q,
        "gold": gold,
        "pred": ans,
        "correct": is_correct
    })

    print(f"\nQ: {q}")
    print(f"Gold: {gold}")
    print(f"Pred: {ans}")
    print(f"Correct? {is_correct}")

# ---------------------------------------------------------
# Print final accuracy
# ---------------------------------------------------------
accuracy = correct / len(qa_data)
print("\n======================")
print(f"KGQA Accuracy: {accuracy:.2f}")
print("======================")

# Save results for inspection
with open("kgqa_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to kgqa_results.json")
