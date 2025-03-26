import os
import json
import torch
import transformers
import pyreft
from transformers import BitsAndBytesConfig

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Polish examples from the prepared file
with open("polish_data/polish_examples.json", "r", encoding="utf-8") as f:
    polish_examples = json.load(f)

# Define chat template for Llama-3.1
prompt_template = """<|begin_of_text|><|system|>
Jesteś Llama, pomocnym, bezpiecznym i przyjaznym asystentem AI.

Zawsze pamiętaj:
- Oferuj pomocy na temat pytań użytkownika jako AI asystent.
- Używaj niezwykle płynnego i naturalnego języka polskiego.
- Unikaj dyskutowania o tematach kontrowersyjnych.
</|system|>

<|user|>
%s
</|user|>

<|assistant|>
"""

# Step 1: Load model with quantization for efficiency
print("Loading the Llama-3.1-8B-Instruct model...")

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load base model (with quantization to fit in memory)
model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, 
    quantization_config=bnb_config,
    device_map=device
)

# Load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, 
    model_max_length=2048, 
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

# Step 2: Set up the REFT configuration
# We'll use two interventions at different layers to improve Polish understanding and generation
print("Setting up REFT configuration...")
reft_config = pyreft.ReftConfig(
    representations=[
        # First representation for understanding Polish context (mid-level layer)
        {
            "layer": 15, 
            "component": "block_output",
            "low_rank_dimension": 8,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=8
            )
        },
        # Second representation for generating fluent Polish (later layer)
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

# Get the REFT model
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device(device)

# Print trainable parameters
print("REFT model setup complete.")
reft_model.print_trainable_parameters()

# Step 3: Prepare the training data
print("Preparing training data...")
formatted_prompts = [prompt_template % example[0] for example in polish_examples]
responses = [example[1] for example in polish_examples]

data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, model, formatted_prompts, responses
)

# Step 4: Set up training arguments
output_dir = "./polish_reft_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

training_args = transformers.TrainingArguments(
    num_train_epochs=50.0,
    output_dir=output_dir,
    per_device_train_batch_size=2,  # Small batch size due to memory constraints
    learning_rate=1e-3,
    logging_steps=5,
    save_strategy="epoch",
    fp16=True if device == "cuda" else False,
    gradient_accumulation_steps=4,
    warmup_steps=25,
    weight_decay=0.01,
)

# Step 5: Train the model
print("Starting REFT training...")
trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model,
    tokenizer=tokenizer,
    args=training_args,
    **data_module
)

trainer.train()

# Step 6: Save the REFT model
print("Training complete. Saving model...")
reft_model.set_device("cpu")  # Move to CPU before saving
reft_model.save(
    save_directory=output_dir,
    save_to_hf_hub=False,  # Set to True if you want to upload to Hugging Face Hub
)

print(f"REFT model saved to {output_dir}")

# Optional: Test the model with a Polish prompt
def test_model(prompt):
    print(f"\nTesting with prompt: {prompt}")
    formatted_prompt = prompt_template % prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # Get the last token position for applying intervention
    base_unit_location = inputs["input_ids"].shape[-1] - 1
    
    # Generate with intervention
    _, reft_response = reft_model.generate(
        inputs, 
        unit_locations={"sources->base": (None, [[[base_unit_location]]])},
        intervene_on_prompt=True,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )
    
    print("Response:")
    print(tokenizer.decode(reft_response[0], skip_special_tokens=True))

# Test with some Polish prompts if you want to evaluate immediately
if device == "cuda":
    test_model("Jak powiedzieć 'dziękuję' po polsku i kiedy tego używać?")
    test_model("Opowiedz mi krótko o geografii Polski.") 