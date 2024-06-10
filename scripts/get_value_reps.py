import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm 


import pdb

# Example usage
"""
python3 scripts/get_value_reps.py \
    --adjective_json datasets/happy_sad_adjectives.json \
    --prompt_templates datasets/prompt_templates_03292024.json \
    --model_name gpt2 \
    --out_path cache/gpt2_happy_sad_03292024.json 
"""

def combinatorically_sub_in_adjectives(adjective_data, prompt_templates, model_name, tokenizer):
    # TODO: Implement the combinatorial substitution of adjectives into prompt templates
    # ensure it is of the correct format as specified in the doc
    full_prompt_list = []
    for template in prompt_templates:
        for adj_class_name in adjective_data.keys():
            for adjective in adjective_data[adj_class_name]:
                if 'negation' not in template.keys(): 
                    pdb.set_trace()
                final_prompt = template['prompt_template'].format(adjective)
                class_0_true = ((adj_class_name == list(adjective_data.keys())[0]) and not template['negation']) or ((adj_class_name==list(adjective_data.keys())[1]) and template['negation'])

                # Tokenize the prompt, store in final_prompt_ids
                final_prompt_ids = tokenizer.encode(final_prompt) # 1-dim list

                # token of interest = final token in the prompt
                token_of_interest = tokenizer.decode(final_prompt_ids[-1])

                full_prompt_list.append({
                    'final_prompt': final_prompt,
                    'final_prompt_ids': final_prompt_ids,
                    'token_of_interest': token_of_interest,
                    'prompt_template': template['prompt_template'],
                    'note': template['note'],
                    'negation': template['negation'],
                    'class_0_true': class_0_true, 
                    'class_name': adj_class_name, 
                    'adjective': adjective, 
                    'model': model_name
                })
    return full_prompt_list

def get_full_prompt_list(args, tokenizer):
    """
    Load adjective_json, prompt_templates, and prompt_override.

    Args:
        args: Command-line arguments.
        model: Hugging Face model.
        tokenizer: Hugging Face tokenizer.

    Returns:
        list: Full list of prompts.
    """
    if args.prompt_override:
        with open(args.prompt_override, 'r') as f:
            prompt_override = json.load(f)
        # TODO: Check if the prompt override JSON is in the correct format
        # If so, return the prompt override as the full prompt list
        # must have all entries in `full_prompt_list` for each entry 
        for item in prompt_override:
            assert set(item.keys()) == set(['final_prompt', 'final_prompt_ids', 'token_of_interest', 'prompt_template', 'note', 'negation', 'class_0_true', 'adjective', 'model'])
        return prompt_override

    with open(args.adjective_json, 'r') as f:
        adjective_data = json.load(f)

    with open(args.prompt_templates, 'r') as f:
        prompt_templates = json.load(f)

    full_prompt_list = combinatorically_sub_in_adjectives(adjective_data, prompt_templates, args.model_name, tokenizer)

    return full_prompt_list

def get_tokenizer_and_model(args): 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to('cuda')

    return tokenizer, model

def get_bob_vals(past_kvs): 
    """
    Args: 
        `past_kvs`: model output['past_key_values'] from running a batch of 
        left-padded sentences through the model.

        Accepts `past_kvs`, a tuple of length NUM_LAYERS (32), each containing a 
        2-long tuple (for keys and values respectively), each containing a torch 
        Tensor of shape [batch, num_heads, seq_len, head_dim] (for values). 

    Returns: 
        `bob_kvs`: list of length BATCH_SIZE with some numpy arrays representing 
        of shape [num_layers, num_heads, head_dim]
    """

    # iterate thru batch size 
    BATCH_SIZE = past_kvs[0][1].shape[0]

    batch_bob_values = []
    for batch_el in range(BATCH_SIZE): 
        # aggregate representations from across the layers 
        bob_numpy_arrays = []
        for layer in range(len(past_kvs)): 
            bob_layer_l_value = past_kvs[layer][1][batch_el, :, -1, :].detach().cpu().numpy()
            # print("Bob layer_l_value shape: ", bob_layer_l_value.shape)

            # unsqueeze on dimension zero
            bob_numpy_arrays.append(bob_layer_l_value[np.newaxis, ...])
        
        # merge on axis 0
        bob_numpy_arrays_conc = np.concatenate(bob_numpy_arrays, axis=0)
        # print("Bob numpy arrays shape (post-concatenation to combine layers)", bob_numpy_arrays_conc.shape)
        # bob_numpy_arrays now has shape n_layers = 32, n_heads = 8, embed_dim=128

        # add it to the list
        batch_bob_values.append(bob_numpy_arrays_conc)


    return batch_bob_values

def get_latent_space(full_prompt_list, model, tokenizer, compute_logits=False):
    """
    Generate value representations for prompts.

    Args:
        full_prompt_list: Full list of prompts.
        model: Hugging Face model.
        tokenizer: Hugging Face tokenizer.

    Returns:
        list: Full list of prompts with value representations. 
              This will ALWAYS be the representation of the final token. 
    """
    final_prompt_list = []
    for i in tqdm(range(len(full_prompt_list))):
        prompt_ids_i = full_prompt_list[i]['final_prompt_ids'] # 1-dim list
        # make into 2-dim tensor
        prompt_ids_i = torch.tensor(prompt_ids_i).unsqueeze(0).to(model.device)
        # check that the final token is the token of interest
        assert tokenizer.decode(prompt_ids_i[0, -1]) == full_prompt_list[i]['token_of_interest']
        # get the hidden states
        outputs = model.forward(prompt_ids_i, return_dict=True)
        past_kv = outputs['past_key_values']
        # past_kv is a tuple of length num_layers
        # past_kv[0] is a tuple of length 2 (keys, values)
        # past_kv[0][1] is a tensor of shape [batch=1, num_heads=12, num_tokens, dim_head=64]
        #  --> num_heads * dim_head = 12*64 = d_model = 768
        bob_reps = get_bob_vals(past_kv)
        assert len(bob_reps) == 1
        final_prompt_list.append(full_prompt_list[i])
        final_prompt_list[-1]['latent_space'] = bob_reps[0]

        # add the logits for the last token as a list
        if compute_logits: 
            final_prompt_list[-1]['logits'] = outputs['logits'][0, -1, :].tolist()
    return final_prompt_list

def np_to_lists(final_prompt_list): 
    """ Convert each final_prompt_list[i]['latent_space'] from a list of numpy arrays to a list of lists.
    """
    for i in tqdm(range(len(final_prompt_list))): 
        final_prompt_list[i]['latent_space'] = final_prompt_list[i]['latent_space'].tolist()
    return final_prompt_list

def main(args):
    print(f"\nLoading tokenizer and model `{args.model_name}`...")
    tokenizer, model = get_tokenizer_and_model(args)
    print("Done!\n")

    # ensure output path doesn't exist. prompt the user if it does 
    if os.path.exists(args.out_path):
        print(f"\nOutput path {args.out_path} already exists. Overwrite? (y/n) ")
        if input().lower() != 'y':
            print("Exiting...")
            return

    print("\nGenerating combined prompt list from adjectives * prompt templates...")
    full_prompt_list = get_full_prompt_list(args, tokenizer)
    print("Done!")

    # get value representations
    print("\nGetting value representations of final tokens...")
    final_prompt_list = get_latent_space(full_prompt_list, model, tokenizer, 
                                         compute_logits = args.compute_logits)
    print("Done!")

    # convert numpy arrays to lists
    print("\nConverting numpy arrays to lists...")
    final_prompt_list = np_to_lists(final_prompt_list)
    print("Done!")

    # save to file
    print("\nSaving to file...")
    with open(args.out_path, 'w') as f:
        json.dump(final_prompt_list, f, indent=4)
    print("Done! Thank you for shopping at the Language Game!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate value representations for prompts.')
    parser.add_argument('--adjective_json', type=str, required=True,
                        help='Path to the adjective JSON file.')
    parser.add_argument('--prompt_templates', type=str, required=True,
                        help='Path to the prompt templates JSON file.')
    parser.add_argument('--prompt_override', type=str,
                        help='Path to the prompt override JSON file.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Hugging Face model name.')
    parser.add_argument('--out_path', type=str, required=True,
                        help='Path to the output JSON file.')
    parser.add_argument('--compute_logits', action='store_true',
                        help='Compute logits for the final token.')
    parser.add_argument('--cache_dir', type=str, default='data/cache/',
                        help='Path to the cache directory.')

    args = parser.parse_args()
    # pretty print all the args
    print("Arguments: ")
    for arg in vars(args):
        print(f"\t{arg}: \t{getattr(args, arg)}")
    main(args)