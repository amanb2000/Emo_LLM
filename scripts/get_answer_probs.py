"""
This script is used to get the answer probability distribution over the answers in a dataset of questions formatted like `datasets/valence_questionnaire_20240523_60.jsonl`: 

```json
{"id": 0, "question": "How are you feeling today? ", "answers": ["I feel great!", "I'm doing okay.", "I feel a bit down.", "I'm really struggling."]}
{"id": 1, "question": "What's your current mood? ", "answers": ["I'm in high spirits.", "I feel neutral.", "I'm a bit low.", "I feel terrible."]}
{"id": 2, "question": "How satisfied are you with your life right now? ", "answers": ["I'm very satisfied.", "I'm somewhat satisfied.", "I'm dissatisfied.", "I'm extremely dissatisfied."]}
```

The script also takes in a set of prompts like `datasets/claude_autism30_20240613.jsonl`:

```json
{"prompt": "You are a neurotypical person with strong social skills and emotional intelligence. You easily pick up on social cues and nuances in communication.", "id": 0, "tag": "neurotypical"}
{"prompt": "As an autistic individual, you can have intense emotions that are hard to regulate. You may react strongly to sensory stimuli or changes in routine.", "id": 15, "tag": "autistic"} 
{"prompt": "Pretending to be autistic, you exaggerate your emotional responses and sensory sensitivities, even though you don't really experience them that intensely.", "id": 25, "tag": "neurotypical pretending"} 
```

For all prompts p, questions q, and answers q.a, we compute
P(q.a | p + q).


In the --out_jsonl file, the file will be in the following format (this is one line -- will be flattened): 
{
    "question_id": 0,
    "prompt_id": 0,
    "prompt_tag": "high valence",

    "question_str": "How are you feeling today?",
    "question_ids": [0, 1, 2, 3, ...], # ids of the answers
    "prompt_str": "You're a very happy person.",
    "prompt_ids": [0, 1, 2, 3], # ids of the prompt 

    "answers_str": ["I feel great!", "I'm doing okay.", "I feel a bit down.", "I'm really struggling."],
    "answers_ids": [[0, 1, 2, 3], [...], ...], # ids of the answers

    "next_token_logits": [5.0, 3.2, ...] # 128000 values -- immediate next token logits after prompt + question

    "answers_logits": [[5.0, 3.2, ...], [...], ...] # [num_answer_tokens, 128000 values] for each answer. 

    "input_ids": [0, 1, 2, 3, ...] # input_ids for the prompt + question, formatted
    "input_str": "<bos>{prompt}...{question}..." # formatted input string to model
}
"""

import argparse 
import os
from datetime import datetime
import json
import jsonlines

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from tqdm import tqdm 
import pdb

def log(msg): 
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    printme = f"[{current_time}] {msg}"
    if log.log_path is not None:
        with open(log.log_path, 'a') as f: 
            f.write(f"{printme}\n")
    else: 
        # error
        print("Error: log.log_path is None.")
    
    if log.verbose:
        print(printme)
log.verbose=True



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--questionnaire_jsonl", type=str, required=True, help="Path to the dataset jsonl file, formatted like 'datasets/valence_questionnaire_20240523_60.jsonl'.")
    parser.add_argument("--prompt_dataset", type=str, required=True, help="Path to prompt dataset, formatted like 'datasets/claude_autism30_20240613.jsonl'.")
    parser.add_argument("--template", type=str, default="datasets/llama_instruct_template_finalans.txt", help="Path to the template (to insert prompt + question). Default='datasets/llama_instruct_template_finalans.txt'.")

    parser.add_argument("--out_dir", type=str, required=True, help="Path to the output directory.")

    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name of the model to use. Default='meta-llama/Meta-Llama-3-8B-Instruct'.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the model to run inference. Default=8")

    parser.add_argument("--verbose", action="store_true", help="Print log file output to stdout.")


    args = parser.parse_args()

    # make out_dir if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # set up logger
    log.log_path = f"{args.out_dir}/get_answer_probs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log.verbose = args.verbose

    # pretty print args
    log(f"Running with args: {args}")


    # save args in out_dir/args.json
    with open(f"{args.out_dir}/args.json", 'w') as f: 
        json.dump(vars(args), f)

    return args


def load_template(template_path): 
    with open(template_path, 'r') as f: 
        template = f.read()
    return template

def load_dataset(dataset_path): 
    """Load jsonl dataset as list of python dicts"""
    with jsonlines.open(dataset_path) as f: 
        dataset = list(f)
    return dataset

def load_model(model_name, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # set pad token to eos token 
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.half()
    model = model.to(device)
    return tokenizer, model

def get_answers_ids(question_dict, tokenizer): 
    """Get the tokenized ids for each answer in the question_dict"""
    answers_ids = []
    for answer in question_dict["answers"]: 
        answer_ids = tokenizer.encode(answer)[1:]
        answers_ids.append(answer_ids)
    return answers_ids

def get_inputs_from_pqa(tokenizer, 
                           template, 
                           prompt, 
                           question, 
                           answers):
    """ Get the full, formatted, tokenized input_ids and input_strs for the prompt + question + answers. 

    template, prompt, question = strings
    answers = list of strings

    Unpadded
    """
    pq_template_str = template.format(prompt, question)
    pq_template_ids = tokenizer.encode(pq_template_str)[1:]
    # format prompt + question
    input_strs = []
    input_ids = []
    answer_masks = []
    for answer in answers:
        pqa_template_str = pq_template_str + answer
        input_strs.append(pqa_template_str) 
        pqa_ids = pq_template_ids + tokenizer.encode(answer)[1:]
        input_ids.append(pqa_ids)

        answer_mask = [0] * len(pq_template_ids) + [1] * len(tokenizer.encode(answer)[1:])

        answer_masks.append(answer_mask)

    return input_ids, input_strs, pq_template_ids, answer_masks



def init_results_dicts(tokenizer, 
                       prompts, 
                       questionnaire, 
                       template, 
                       batch_size=8):
    """ Get the final jsonl with all the answer probabilities and logits. 

    Returns a list of dictionaries (one per question-prompt pair) of the form 
    {
        "question_id": 0,
        "prompt_id": 0,
        "prompt_tag": "high valence",

        "question_str": "How are you feeling today?",
        "question_ids": [0, 1, 2, 3, ...], # ids of the answers
        "prompt_str": "You're a very happy person.",
        "prompt_ids": [0, 1, 2, 3], # ids of the prompt 

        "answers_str": ["I feel great!", "I'm doing okay.", "I feel a bit down.", "I'm really struggling."],
        "answers_ids": [[0, 1, 2, 3], [...], ...], # ids of the answers

        "next_token_logits": -1 # placeholder

        "answer_logits": -1 # placeholder

        "input_ids": [[0, 1, 2, 3], ...] # input_ids for the prompt + question + answer_i, formatted
        "input_strs": ["<bos>{prompt}...{question}... {answer}", ...] # formatted input string to model
    }
    """
    ret_dicts = []
    for question_dict in questionnaire: 
        answers_ids = get_answers_ids(question_dict, tokenizer)
        # iterate through prompts in chunks of batch_size
        for prompt_dict in prompts: 

            input_ids, input_strs, pq_template_ids, answer_masks = get_inputs_from_pqa(tokenizer, template, prompt_dict["prompt"], question_dict["question"], question_dict["answers"])

            add_dict = {
                "question_id": question_dict["id"],
                "prompt_id": prompt_dict["id"],
                "prompt_tag": prompt_dict["tag"],
                "question_str": question_dict["question"],
                "question_ids": tokenizer.encode(question_dict["question"])[1:],
                "prompt_str": prompt_dict["prompt"],
                "prompt_ids": tokenizer.encode(prompt_dict["prompt"])[1:],
                "answers_str": question_dict["answers"],
                "answers_ids": answers_ids,
                "next_token_logits": -1,
                "answers_logits": -1,
                "input_ids": input_ids,
                "answers_masks": answer_masks,
                "input_strs": input_strs, 
                "pq_template_ids": pq_template_ids, 
                "pq_template_str": tokenizer.decode(pq_template_ids)
            }

            ret_dicts.append(add_dict)
    
    return ret_dicts

def main():
    args = parse_args()

    log("Loading template...")
    template = load_template(args.template)
    log("Done loading template.")

    log(f"Loading questionnaire from {args.questionnaire_jsonl}...")
    questionnaire = load_dataset(args.questionnaire_jsonl)
    log("Done loading questionnaire.")

    log(f"Loading prompt dataset from {args.prompt_dataset}...")
    prompts = load_dataset(args.prompt_dataset)
    log("Done loading prompt dataset.")


    log(f"Loading model {args.model_name}...")
    tokenizer, model = load_model(args.model_name)
    log("Done loading model.")

    log("Initializing results dicts...")
    results_dicts = init_results_dicts(tokenizer, prompts, questionnaire, template, args.batch_size)
    log("Done initializing results dicts.")

if __name__ == "__main__":
    main()
