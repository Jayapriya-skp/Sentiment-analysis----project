import torch

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    AutoTokenizer,
)
from transformers.pipelines.pt_utils import KeyDataset

from datasets import load_dataset

from social_media_nlp.models.transformers.inference import load_language_model
from social_media_nlp.models.transformers.utils import (
    smart_tokenizer_and_embedding_resize,
)
from social_media_nlp.models.utils import output2label
from social_media_nlp.models.evaluation import compute_classification_metrics
from social_media_nlp.prompts import MODEL_PROMPT_TEMPLATE, SA_TEMPLATE_COMPLETION

import json

import argparse
import os

import tqdm
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model to evaluate")

    args = parser.parse_args()

    model_path = str(args.model_path)

    # Load model
    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = load_language_model(
        model_path,
        model_args={
            "quantization_config": bnb_config,
            "low_cpu_mem_usage": True,
            "attn_implementation": "flash_attention_2",
        },
        merged_model=False,
    )

    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        # padding_side="right",
        padding_side="left",
        trust_remote_code=True,
    )
    special_tokens_dict = {"pad_token": tokenizer.eos_token}
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    # Generation parameters
    max_new_tokens = 3 if "Phi" in model_path else 5

    generation_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.1,
    }

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        **generation_params,
    )

    # Load dataset
    dataset = load_dataset("tweet_eval", "sentiment")

    base_model_name = "".join(model_path.split("_")[:2])
    base_model_name = "/".join(base_model_name.split("/")[-3:-1])
    model_prompt = MODEL_PROMPT_TEMPLATE.get(base_model_name, "{prompt}")

    test_dataset = dataset["test"].map(
        lambda entry: {
            "prompt": model_prompt.format(
                prompt=SA_TEMPLATE_COMPLETION.format(examples="", text=entry["text"])
            ),
        }
    )

    # Evaluate
    start_time = time.time()

    result = [
        out
        for out in tqdm.tqdm(
            pipe(KeyDataset(test_dataset, "prompt"), batch_size=8, pad_token_id=2)
        )
    ]

    end_time = time.time()
    execution_time = end_time - start_time

    predictions = list(
        map(
            lambda x: output2label(x[0]["generated_text"].strip()),
            result,
        )
    )

    metrics = compute_classification_metrics(
        predictions,
        dataset["test"]["label"],
    )

    save_dir = f"./models/tweet_eval/predictions/tweet_eval/{base_model_name}/{model_path.split('/')[-1]}/"
    os.makedirs(save_dir, exist_ok=True)

    with open(
        save_dir + "predictions.json",
        "w",
    ) as f:
        json.dump(
            {
                "raw_predictions": result,
                "predictions": predictions,
                "execution_time": execution_time,
                "metrics": metrics,
            },
            f,
        )


if __name__ == "__main__":
    main()
