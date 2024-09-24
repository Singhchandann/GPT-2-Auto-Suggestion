# GPT-2-Auto-Suggestion
This project trains a GPT-2 model from scratch for auto-suggestion tasks on custom data. The model can be used to suggest the next words in multilingual based on a given prefix.  

## Project Structure
- `data/`: Contains the custom training data.
- `src/`: Code for model training and utility functions.
- `tokenizer/`: Folder where trained tokenizer is saved.
- `output/`: Folder where model checkpoints are saved.
- `logs/`: Folder for training logs.
- `.gitignore`: Ignore unnecessary files.
- `requirements.txt`: List of Python dependencies.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Hugging Face Transformers

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python src/train.py
```
### License  
This project is licensed under the MIT License - see the LICENSE file for details.
