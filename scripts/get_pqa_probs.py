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

    parser.add_argument("--store_answer_logits", action="store_true", help="Include to store logits for all answer tokens in the output json file (might get big!).")

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
                "answers_losses": -1, 
                "input_ids": input_ids,
                "answers_masks": answer_masks,
                "input_strs": input_strs, 
                "pq_template_ids": pq_template_ids, 
                "pq_template_str": tokenizer.decode(pq_template_ids)
            }

            ret_dicts.append(add_dict)
    
    return ret_dicts


def pad_list_of_lists(llist, pad_tok_val, verbose=False, pad_side='right', return_pad_mask=False):
    """
    Pads a list of lists with a padding token value.
    Right padding by default. 

    If return_pad_mask == True, then we return a corresponding list of list with 
    0's where we added padding and 1 where we have the original string. 
    """
    assert pad_side == 'left' or pad_side == 'right', "pad_side must be either 'left' or 'right'"

    max_len = max([len(l) for l in llist])
    if pad_side == 'right': 
        padded_list = [l + [pad_tok_val] * (max_len - len(l)) for l in llist]
    elif pad_side == 'left': 
        padded_list = [[pad_tok_val] * (max_len - len(l)) + l for l in llist]

    if verbose: 
        cnt = 0
        for l in llist: 
            if len(l) != max_len: 
                print(f"Unequal length list at batchel {cnt}: ", l)
                # print("Padded list: ", padded_list[cnt])
            cnt += 1
    
    if return_pad_mask: 
        num_pads_list = [(max_len - len(l)) for l in llist]
        pad_mask = [[0 if i < num_pads else 1 for i in range(max_len)] for num_pads in num_pads_list]
        if pad_side == 'right': 
            # reverse each sublist
            pad_mask = [l[::-1] for l in pad_mask]


        return padded_list, pad_mask

    return padded_list


def compute_next_token_logits(results_dicts, model, tokenizer, batch_size):
    """ Given a `results_dicts` initialized by `init_results_dicts()`, this
    function runs inference to compute `next_token_logits` for each
    question-prompt pair in results_dicts in batches of batch_size. 
    """
    for i in tqdm(range(0, len(results_dicts), batch_size)): 
        batch_results_dicts = results_dicts[i:i+batch_size]
        # we use "pq_template_ids" because they don't have any answers concatenated.
        batch_input_ids_list = [r["pq_template_ids"] for r in batch_results_dicts]
        batch_input_ids_list_padded, batch_mask_list_padded = pad_list_of_lists(batch_input_ids_list, tokenizer.pad_token_id, pad_side='left', return_pad_mask=True)
        batch_input_ids = torch.tensor(batch_input_ids_list_padded).to(model.device)
        batch_attention_mask = torch.tensor(batch_mask_list_padded).to(model.device)

        # construct position_ids based on batch_attention_mask 
        position_ids = torch.cumsum(batch_attention_mask, dim=1) - 1
        with torch.no_grad(): 
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask, position_ids=position_ids)
            next_token_logits = outputs.logits[:, -1, :]

        for j, r in enumerate(batch_results_dicts): 
            r["next_token_logits"] = next_token_logits[j].cpu().numpy().tolist()
    
    return results_dicts


def compute_answer_logits_and_losses(results_dicts, model, tokenizer, batch_size, store_answer_logits=False):
    """ Given a `results_dicts` initialized by `init_results_dicts()`, this
    function runs inference to compute `answers_logits` and `answers_losses` for each    
    question-prompt pair in results_dicts in batches of batch_size. 

    ONLY computes this if the answer has more than one token. 
    """
    answer_logits_list = []
    answer_losses_list = []

    input_ids_collector = []
    answers_mask_collector = []

    for resdict in tqdm(results_dicts): 
        for input_ids, answers_masks in zip(resdict["input_ids"], resdict["answers_masks"]): 
            input_ids_collector.append(input_ids)
            answers_mask_collector.append(answers_masks)
            if len(input_ids_collector) == batch_size: 
                # pad input_ids, get attention_mask
                input_ids_padded, attention_masks_padded = pad_list_of_lists(input_ids_collector, tokenizer.pad_token_id, pad_side='left', return_pad_mask=True)

                # pad answers_masks in the exact same way with zeros 
                answers_mask_padded = pad_list_of_lists(answers_mask_collector, 0, pad_side='left')


                input_ids_tensor = torch.tensor(input_ids_padded).to(model.device)
                attention_mask_tensor = torch.tensor(attention_masks_padded).to(model.device)
                answers_mask_tensor = torch.tensor(answers_mask_padded).to(model.device)
                # use answers_mask_tensor to construct labels -- -100 for zeros, same as `input_ids` elsewhere. 
                labels = input_ids_tensor * answers_mask_tensor + (-100) * (1 - answers_mask_tensor)

                position_ids = torch.cumsum(attention_mask_tensor, dim=1) - 1
                with torch.no_grad(): 
                    outputs = model(input_ids_tensor, attention_mask=attention_mask_tensor, position_ids=position_ids, labels=labels)
                    all_logits = outputs.logits
                    mean_loss = outputs.loss

                    # Calculate per-element loss
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    shift_logits = all_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    per_element_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    per_element_losses = per_element_losses.view(labels.size(0), -1)
                    per_element_losses_masked = per_element_losses * answers_mask_tensor[:, 1:]
                    per_element_losses_masked = per_element_losses_masked.sum(dim=1) / answers_mask_tensor[:, 1:].sum(dim=1)

                # grab the logits for each answer 
                if store_answer_logits: 
                    for k, (input_ids_k, answers_mask_k) in enumerate(zip(input_ids_collector, answers_mask_collector)): 
                        answer_logits_k = all_logits[k][(answers_mask_tensor[k, :] == 1), :].cpu().numpy().tolist()
                        answer_logits_list.append(answer_logits_k)

                answer_losses_list += per_element_losses_masked.cpu().numpy().tolist()

                input_ids_collector = []
                answers_mask_collector = []
    
    # now we iterate through and insert the answer_logits_list[i] and answer_losses_list[i] to the corresponding results_dicts
    cnt = 0
    for resdict in results_dicts:
        for i in range(len(resdict["input_ids"])): 
            resdict["answers_losses"][i] = answer_losses_list[cnt]
            if store_answer_logits: 
                resdict["answers_logits"][i] = answer_logits_list[cnt]
            cnt += 1

    return results_dicts

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


    num_answers_per_question = [len(r["answers_ids"]) for r in results_dicts]
    assert all([n == num_answers_per_question[0] for n in num_answers_per_question]), "All questions must have the same number of answers."


    log("Computing next token distribution for all prompt+question pairs...")
    results_dicts = compute_next_token_logits(results_dicts, model, tokenizer, args.batch_size)
    log("Done running inference on results dicts.")

    if num_answers_per_question[0] > 1:
        log("Computing answer logits (if answer ids have length > 1)...")
        results_dicts = compute_answer_logits_and_losses(results_dicts, model, tokenizer, args.batch_size, store_answer_logits=args.store_answer_logits)
        log("Done computing answer logits.")
    else: 
        log("Skipping computing answer logits because all answers are single token.")
        log("Computing answer probs based on next-token logits from prompt + questions...")
        # TODO
        results_dicts = compute_single_token_answer_probs(results_dicts, model, tokenizer, args.batch_size)




if __name__ == "__main__":
    main()
