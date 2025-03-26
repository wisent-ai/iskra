# Llama 3.1 Polish Enhancement with REFT

This project uses [REFT (Representation Fine-Tuning)](https://github.com/stanfordnlp/pyreft) to improve the Polish language capabilities of Meta's Llama-3.1-8B-Instruct model.

## What is REFT?

REFT (Representation Fine-Tuning) is a parameter-efficient fine-tuning method that applies interventions at specific layers of a large language model to modify its behavior. Unlike traditional fine-tuning or PEFT (Parameter-Efficient Fine-Tuning) methods like LoRA that operate on weights across all timesteps, REFT:

1. Selects specific timesteps to intervene on
2. Targets representations rather than weights
3. Requires very few trainable parameters
4. Can be trained with a small number of examples

This makes REFT ideal for enhancing language models for specific languages or tasks with minimal computational resources.

## Project Structure

- `prepare_polish_data.py`: Creates a dataset of Polish instruction-response pairs
- `train_polish_reft.py`: Trains Llama-3.1-8B-Instruct with REFT for Polish language improvement
- `evaluate_polish_reft.py`: Compares the original and REFT-enhanced models on Polish tasks
- `polish_data/`: Directory containing the Polish training examples
- `polish_reft_model/`: Directory containing the saved REFT model after training
- `evaluation_results/`: Directory containing evaluation results

## How It Works

1. **Data Preparation**: We create a dataset of diverse Polish instruction-response pairs covering grammar, vocabulary, cultural content, and more.

2. **REFT Configuration**: We configure two REFT interventions at different layers:
   - Layer 15: For improving Polish language understanding
   - Layer 25: For improving Polish language generation

3. **Training**: We train the REFT model on our Polish examples, which requires only a fraction of the parameters of the full model.

4. **Evaluation**: We compare the original and REFT-enhanced models on various Polish language tasks.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- REFT (pyreft)
- CUDA-capable GPU (recommended)

Install the required packages:

```bash
pip install pyreft transformers torch huggingface-hub datasets accelerate sentencepiece
```

## Usage

### 1. Prepare Polish Data

```bash
python prepare_polish_data.py
```

This creates a dataset of Polish examples in `polish_data/polish_examples.json`.

### 2. Train REFT Model

```bash
python train_polish_reft.py
```

This trains the REFT model and saves it to `polish_reft_model/`.

### 3. Evaluate the Model

```bash
python evaluate_polish_reft.py
```

This compares the original and REFT-enhanced models on Polish tasks and saves the results to `evaluation_results/comparison_results.json`.

## How to Use the Enhanced Model

The REFT-enhanced model can be loaded and used for Polish language tasks as follows:

```python
import torch
import transformers
import pyreft

# Load the base model
model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", 
    device_map="auto"
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    model_max_length=2048,
    padding_side="right"
)

# Create REFT configuration (must match the training configuration)
reft_config = pyreft.ReftConfig(
    representations=[
        {
            "layer": 15, 
            "component": "block_output",
            "low_rank_dimension": 8,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=8
            )
        },
        {
            "layer": 25, 
            "component": "block_output",
            "low_rank_dimension": 8,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=8
            )
        }
    ]
)

# Load the REFT model
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.load("./polish_reft_model")

# Use the model
prompt = "Jak opisać polską kulturę w kilku zdaniach?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
base_unit_location = inputs["input_ids"].shape[-1] - 1

_, reft_response = reft_model.generate(
    inputs, 
    unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7
)

print(tokenizer.decode(reft_response[0], skip_special_tokens=True))
```

## Advantages of Using REFT

1. **Resource Efficiency**: Trains with minimal parameters (~32K vs billions in the full model)
2. **Data Efficiency**: Requires only a small number of examples (10-20 can be sufficient)
3. **Targeted Improvement**: Specifically enhances Polish language understanding and generation
4. **Preservation of Original Capabilities**: Maintains the model's capabilities in other languages

## Limitations

1. The enhancements are focused on Polish language capabilities only
2. Performance depends on the quality and diversity of the training examples
3. The REFT approach might not be as comprehensive as full fine-tuning with large datasets

## License

This project uses the same license as the base Llama 3.1 model. Please refer to Meta's licensing terms for the Llama model.
