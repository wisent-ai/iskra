import torch
import transformers
import pyreft
import time
import json
import os
from transformers import BitsAndBytesConfig

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Llama-3.1 chat template
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

# Evaluation prompts in Polish that focus on different linguistic aspects
evaluation_prompts = [
    "Wyjaśnij, jak działa odmiana rzeczowników w języku polskim. Podaj przykłady dla różnych przypadków.",
    "Napisz krótkie opowiadanie o tradycyjnym polskim weselu.",
    "Porównaj język polski z innymi językami słowiańskimi. Jakie są główne podobieństwa i różnice?",
    "Wytłumacz, jak używać przyimków 'w' oraz 'na' w języku polskim. Kiedy używamy jednego, a kiedy drugiego?",
    "Przetłumacz następujące zdanie na polski: 'The beautiful sunset painted the sky with vibrant colors of orange and pink.'",
    "Jakie są najtrudniejsze aspekty gramatyki polskiej dla obcokrajowców? Wyjaśnij dlaczego.",
    "Napisz dialog między dwoma osobami spotykającymi się w kawiarni po polsku, używając potocznego języka."
]

# Load the original model
print("\nLoading the original Llama-3.1-8B-Instruct model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

original_model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map=device
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", 
    model_max_length=2048, 
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

# Load the REFT-enhanced model
print("\nLoading the REFT-enhanced Polish model...")
reft_output_dir = "./polish_reft_model"

# Check if the REFT model exists
if not os.path.exists(reft_output_dir):
    print(f"Error: REFT model not found at {reft_output_dir}. Please train the model first.")
    exit(1)

# Load the REFT configuration
reft_config = pyreft.ReftConfig(
    representations=[
        # First representation for understanding Polish context (mid-level layer)
        {
            "layer": 15, 
            "component": "block_output",
            "low_rank_dimension": 8,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=original_model.config.hidden_size,
                low_rank_dimension=8
            )
        },
        # Second representation for generating fluent Polish (later layer)
        {
            "layer": 25, 
            "component": "block_output",
            "low_rank_dimension": 8,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=original_model.config.hidden_size,
                low_rank_dimension=8
            )
        }
    ]
)

# Get the REFT model with the same configuration
reft_model = pyreft.get_reft_model(original_model, reft_config)
# Load the saved weights
reft_model.load(reft_output_dir)
reft_model.set_device(device)

# Function to generate responses from models
def generate_response(model, prompt, is_reft=False):
    formatted_prompt = prompt_template % prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    
    if is_reft:
        # Get the last token position for applying intervention
        base_unit_location = inputs["input_ids"].shape[-1] - 1
        
        # Generate with intervention
        _, response_ids = model.generate(
            inputs, 
            unit_locations={"sources->base": (None, [[[base_unit_location]]])},
            intervene_on_prompt=True,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id
        )
    else:
        # Regular generation for original model
        response_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    
    # For REFT model, we need to extract just the assistant's response
    if formatted_prompt in response_text:
        response_text = response_text.split(formatted_prompt)[-1].strip()
    
    return response_text, generation_time

# Create results directory
if not os.path.exists("evaluation_results"):
    os.makedirs("evaluation_results")

# Run the evaluation
print("\nRunning evaluation...")
results = []

for i, prompt in enumerate(evaluation_prompts):
    print(f"\nEvaluating prompt {i+1}/{len(evaluation_prompts)}")
    print(f"Prompt: {prompt}")
    
    # Generate response from original model
    print("Generating response from original model...")
    original_response, original_time = generate_response(original_model, prompt, is_reft=False)
    
    # Generate response from REFT model
    print("Generating response from REFT-enhanced model...")
    reft_response, reft_time = generate_response(reft_model, prompt, is_reft=True)
    
    result = {
        "prompt": prompt,
        "original_model": {
            "response": original_response,
            "generation_time": original_time
        },
        "reft_model": {
            "response": reft_response,
            "generation_time": reft_time
        }
    }
    
    results.append(result)
    
    print("\nOriginal model response:")
    print(original_response)
    print(f"Generation time: {original_time:.2f} seconds")
    
    print("\nREFT-enhanced model response:")
    print(reft_response)
    print(f"Generation time: {reft_time:.2f} seconds")

# Save results to file
with open("evaluation_results/comparison_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nEvaluation complete! Results saved to evaluation_results/comparison_results.json")
print("\nCompare the responses to see how the REFT-enhanced model performs on Polish language tasks.") 