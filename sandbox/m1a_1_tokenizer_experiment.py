from transformers import AutoTokenizer,PreTrainedTokenizerBase
from pathlib import Path
import re
import random
import wandb
from text_utils import *



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
