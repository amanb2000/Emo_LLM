# %%
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")# %%

# %%


import re
def get_eos_tokens(tokenizer, verbose=False): 
    """ Given a tokenizer, return a list of all the token ids that 
    have a period, question mark, or exclamation point.
    """
    # Get the token strings
    token_strs = []
    for i in range(tokenizer.vocab_size): 
        token_strs.append(tokenizer.decode([i]))

    # Look for 
    eos_token_ids = []
    pattern = r"[.!?]"
    for idx, st in enumerate(token_strs): 
        matches = re.findall(pattern, st)
        if matches: 
            if verbose: 
                print(f'{idx}: ', st)
            eos_token_ids.append(idx)
    
    return eos_token_ids

    # %%
