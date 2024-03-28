# Emotional LLMs 

## Setup

```bash
# create a virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install the package and dependencies
pip install -e .
pip install -r requirements.txt
```

## Repo 
 - `scripts/`: Code for doing large-scale experiments analyzing the robustness 
 of the internal representational geometry of LLMs w.r.t. emotional valence/etc. 
     - `get_value_reps.py`: Given a set of adjectives and prompt templates, 
     output a JSON object with all the latent representations corresponding 
     to the final token in the sequence. 
     - `grok_intrinsic_geometry.py`: [TODO: CAYDEN](https://caydenpierce.com/)
 - `datasets/`: Adjective lists, prompt templates, etc. 
 - `notebooks/`: Prototype Jupyter notebooks for analyzing latent 
 representations of transformer-based LLMs from HuggingFace. 
 - `html/`: Plotly interactive plots of PCA on representations. 
