import streamlit as st
import os
import time
import socket
import uuid
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import matplotlib.pyplot as plt
import hashlib

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# Function to load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.half()
    model = model.to('cuda')
    return tokenizer, model

# Function to get the client's IP address
def get_client_ip():
    return socket.gethostbyname(socket.gethostname())

# Function to get or create a session ID
def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log(msg, file_path="games/log.questionnaire_design"): 
    with open(file_path, 'a') as f:
        f.write(f"[{datetime.now()}] {msg}\n")

def plot_next_tok_dist(input_str, tokenizer, model, session): 
    hashed_input_str = hashlib.md5(input_str.encode()).hexdigest()
    log(f"(Session {session}) Received input_str = {input_str}")
    log(f"(Session {session}) Hashed input_str: {hashed_input_str}")
    input_ids = tokenizer.encode(input_str, return_tensors="pt").to(model.device)
    recovered_input_str = tokenizer.decode(input_ids[0])
    log(f"(Session {session}) Recovered input_str = {recovered_input_str}")

    cache_path = f"games/logs/top100_{hashed_input_str}.csv"
    if os.path.exists(cache_path):
        log(f"(Session {session}) Cache found, loading from cache")
        df = pd.read_csv(cache_path)
        log(f"(Session {session}) Done loading from cache")
        # Create a figure and axis
    else: 
        # run the model 
        log(f"(Session {session}) Running model for input_str = {input_str}")
        with torch.no_grad():
            output = model(input_ids)
        log(f"Done running model for input_str = {input_str} (Session {session})")

        # we are interested in the estimate of the next token after the end of the string
        next_token_probs = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_probs, dim=0)

        next_token_dict = {}

        for i in range(next_token_probs.shape[0]): 
            next_token_dict[tokenizer.decode(i)] = next_token_probs[i].item()

        df = pd.DataFrame(next_token_dict.items(), columns=['token', 'probability'])
        df = df.sort_values(by='probability', ascending=False)
        df = df.reset_index(drop=True)
        # save to disk at cache path 
        log(f"(Session {session}) Saving to cache at {cache_path}")
        df.head(100).to_csv(cache_path)
        log(f"(Session {session}) Done saving to cache")


    # Create a figure and axis
    st.write(f"## Next token probability distribution for input string: `{input_str}`")
    log(f"(Session {session}) Plotting...")
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plotting on the specified axis
    df.head(100).plot(kind='bar', x='token', y='probability', ax=ax)
    if len(input_str) > 100: 
        ax.set_title("Input str: " + input_str[:100] + "[...]")
    else: 
        ax.set_title("Input str: " + input_str)

    # Show the plot in Streamlit
    st.pyplot(fig)

    # show the dataframe head 100 
    st.write(df.head(100))
    log(f"(Session {session}) Done plotting")

# Main Streamlit app
def main():
    st.title("Next Token Widget")
    ip_file = "games/log.questionnaire_design"
    session_file = "games/session.questionnaire_design"
    log_file = "games/log.questionnaire_design"
    current_session_id = get_session_id()

    log(f"Access by user at IP {get_client_ip()} at session id {current_session_id}")

    # write the current session ID and the current timestamp to a file
    with open(session_file, 'a') as f:
        f.write(f"Session ID: {current_session_id}\nDate: {datetime.now()}\n")

    # title: Next token probability scores
    st.write("Enter a string and find the probability distribution over the next token!")
    # get user input for input_string
    input_string = st.text_input("Input string")

    # button 
    if st.button("Run model"):
        plot_next_tok_dist(input_string, tokenizer, model, current_session_id)

# Load the model and tokenizer once
tokenizer, model = load_model()

if __name__ == "__main__":
    main()
