from transformers import AutoTokenizer,PreTrainedTokenizerBase
from pathlib import Path
import re
import random
import wandb
para_splitter = "\n\n" 
FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)   # block between the first two '---' lines
COMMENT      = re.compile(r"<!--.*?-->", re.S)                 # any HTML comment, also multi-line
BLANK_LINE   = re.compile(r"\n\s*\n")                          # paragraph separator
WHITESPACE   = re.compile(r"\s+")
ONLY_NUMBER  = re.compile(r"\d+")                              # sidebar numbers like '1' ... '16'
MODEL_DICT = {
    "e5" : "intfloat/multilingual-e5-large",
    "gemma" : "google/embeddinggemma-300m",
    "bge" : "BAAI/bge-m3",
    "berturk" : "trmteb/turkish-embedding-model",
    "qwen_06" : "Qwen/Qwen3-Embedding-0.6B",
    "labse" : "sentence-transformers/LaBSE",
    
}
MODEL_DICT_MAX_TOKEN = {
    "e5": 512,          # model card: "Long texts will be truncated to at most 512 tokens"
    "gemma": 2048,      # model card: "Maximum input context length of 2048 tokens"
    "bge": 8192,        # model card: "up to 8192 tokens"
    "berturk": 512,     # NOT verified on the model card: inferred from the transformers warning (865 > 512)
    "qwen_06": 32768,   # model card says "32k"; 32768 assumed as the exact value
    "labse": 512,       # NOT verified on the model card: inferred from the transformers warning (809 > 512)
}
def clean_text(text: str) -> str:
    """Remove front matter and comments; keep the document text and its blank lines."""
    text = FRONT_MATTER.sub("", text, count=1)
    text = COMMENT.sub("", text)
    return text

def split_paragraphs(text: str, keep_headings: bool = False, min_chars: int = 3) -> list[str]:
    """Split cleaned markdown into paragraphs (one string each, whitespace collapsed)."""
    paragraphs = []
    for piece in BLANK_LINE.split(text):
        p = WHITESPACE.sub(" ", piece).strip()
        if not p:
            continue                                  # empty piece
        if p.startswith("#") and not keep_headings:
            continue                                  # markdown heading
        if ONLY_NUMBER.fullmatch(p):
            continue                                  # sidebar number
        if len(p) < min_chars:
            continue                                  # leftover fragment
        paragraphs.append(p)
    return paragraphs



def load_paragraphs(path: Path, **kwargs) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return split_paragraphs(clean_text(text), **kwargs)




def measure_tokenizer(paras,model_name, model_link):
    """
    Given a list of paragraphs, this function calculates the average number of characters that a single token contains for an embeding model.

    Note that embeddin models that this function uses are determined via MODEL_DICT dictionary.

    """
    model_token_dict = {}
    model_token_limit = MODEL_DICT_MAX_TOKEN[model_name] 

    tokenizer = AutoTokenizer.from_pretrained(model_link)
    tok = tokenizer(paras,add_special_tokens = False)["input_ids"]

    
    paras_chars = list(map(len,paras))
    paras_tokens =  [len(ids) for ids in tok]

    total_chars = sum(paras_chars)
    total_tokens = sum(paras_tokens)
    
    avg_token = total_chars/total_tokens
    model_token_dict[model_name] = avg_token
    return {"model_short_name" : model_name,
        "avg_chars_per_token": avg_token ,
            "total_tokens_for_doc": total_tokens ,
            "longest_paragraph_tokens": max(paras_tokens),
            "token_margin" : (model_token_limit - max(paras_tokens)),
            "n_paragraphs_over_limit" : sum([i > model_token_limit for i in paras_tokens ])
            }

if __name__ == "__main__":
    
    REPO_ROOT = Path(__file__).resolve().parents[1]   # sandbox/ -> repo root
    path = REPO_ROOT / "data" / "text" / "doc01_yapay_zeka_eylem_plani.md"
    if not path.exists() :
            raise FileNotFoundError(f"The document does not exist on the path provided {path}")
    paras = load_paragraphs(path)

    for model_name, model_link in MODEL_DICT.items():
        with wandb.init(project="rag-eval-lab", name=f"m1a-{model_name}", group="m1a-tokenizers", config={"model_name"  : model_name,
                                                                                                           "model_link"  : model_link,
                                                                                                            "max_token" : MODEL_DICT_MAX_TOKEN[model_name],
                                                                                                           "document" : "Yapay_zeka_eylem_plani",
                                                                                                           "no_of_paragraph" : len(paras),
                                                                                                           "language" : "Turkish",
                                                                                                           }) as run:
        

            tokenizer_dict = measure_tokenizer(paras,model_name,model_link)
            for key,value in tokenizer_dict.items():
                run.summary[key] = value
