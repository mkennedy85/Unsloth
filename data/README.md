# Dataset Information

This document describes the datasets used in each colab notebook and how to prepare your own.

## Colab 1 & 2: Instruction Fine-Tuning Datasets

### Primary Dataset: Alpaca Cleaned
- **HuggingFace**: `yahma/alpaca-cleaned`
- **Size**: 52,000 instruction-response pairs
- **Format**: JSON with instruction/input/output fields
- **Task**: General instruction following
- **Quality**: Cleaned version of original Alpaca dataset (removed incomplete/incorrect examples)

**Example:**
```json
{
  "instruction": "Give three tips for staying healthy.",
  "input": "",
  "output": "1. Eat a balanced diet. 2. Exercise regularly. 3. Get enough sleep."
}
```

### Alternative Datasets for Colabs 1 & 2

#### 1. Coding Tasks
**Dataset**: `iamtarun/python_code_instructions_18k_alpaca`
- 18k Python coding instructions
- Great for code generation tasks
```json
{
  "instruction": "Write a Python function to reverse a string",
  "input": "",
  "output": "def reverse_string(s):\n    return s[::-1]"
}
```

**Dataset**: `TokenBender/code_instructions_122k_alpaca_style`
- 122k multi-language code instructions
- Covers Python, JavaScript, Java, C++, etc.

#### 2. Chat/Conversation
**Dataset**: `OpenAssistant/oasst1`
- Real human-AI conversations
- Multi-turn dialogues
- 161k messages

**Dataset**: `HuggingFaceH4/ultrachat_200k`
- High-quality synthetic conversations
- Diverse topics and styles

#### 3. Domain-Specific

**Medical**: `medalpaca/medical_meadow_medqa`
- Medical question-answering
- 10k+ medical exam questions

**Legal**: `pile-of-law/pile-of-law`
- Legal documents and analysis
- Various legal tasks

**Math**: `competition_math`
- Math problem solving
- Step-by-step solutions

#### 4. Small Datasets (for quick testing)
**Dataset**: `tatsu-lab/alpaca` (first 1000 examples)
```python
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
```

## Colab 3: Preference Learning (DPO/ORPO)

### Required Format
Datasets for reinforcement learning need **chosen** and **rejected** responses:

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris, one of the most beautiful cities in Europe.",
  "rejected": "France capital is Paris."
}
```

### Recommended Datasets

#### 1. Anthropic HH-RLHF
**Dataset**: `Anthropic/hh-rlhf`
- 169k human preference comparisons
- Helpful and harmless dialogues
- Gold standard for RLHF

```python
dataset = load_dataset("Anthropic/hh-rlhf")
print(dataset['train'][0])
# Output: {'chosen': '...', 'rejected': '...'}
```

#### 2. UltraFeedback
**Dataset**: `HuggingFaceH4/ultrafeedback_binarized`
- 64k preference pairs
- Multiple quality dimensions
- Pre-processed for DPO

#### 3. Helpful-Harmless (HH)
**Dataset**: `HuggingFaceH4/stack-exchange-preferences`
- Stack Exchange voting data
- Technical Q&A preferences
- 10M+ pairs

#### 4. Code Preferences
**Dataset**: `lvwerra/stack-exchange-paired`
- Code snippet preferences
- Based on upvotes/downvotes

### Creating Your Own Preference Dataset

```python
# Example structure
data = [
    {
        "prompt": "Explain recursion",
        "chosen": "Recursion is when a function calls itself to solve smaller instances of the same problem. Base case prevents infinite loops.",
        "rejected": "Recursion is a programming thing where functions call themselves."
    },
    # ... more examples
]

from datasets import Dataset
dataset = Dataset.from_list(data)
```

## Colab 4: GRPO (Reasoning Model)

### Required Format
GRPO needs problems where the model generates multiple solutions:

```json
{
  "problem": "If a train travels 120 km in 2 hours, what is its average speed?",
  "solution_steps": [
    "Step 1: Identify given values",
    "Step 2: Apply formula: speed = distance / time",
    "Step 3: Calculate: 120 km / 2 hours = 60 km/h",
    "Step 4: Answer: 60 km/h"
  ]
}
```

### Recommended Datasets

#### 1. GSM8K (Math)
**Dataset**: `gsm8k`
- 8.5k grade school math problems
- Step-by-step solutions
- Perfect for reasoning training

```python
dataset = load_dataset("gsm8k", "main")
print(dataset['train'][0])
# Output: {'question': '...', 'answer': 'Step 1: ...\nStep 2: ...'}
```

#### 2. MATH Dataset
**Dataset**: `competition_math`
- Competition-level math
- Multiple difficulty levels
- Detailed solutions

#### 3. OpenOrca Reasoning
**Dataset**: `Open-Orca/OpenOrca`
- 1M+ reasoning examples
- Diverse reasoning tasks
- GPT-4 generated explanations

#### 4. ARC Challenge
**Dataset**: `ai2_arc` (challenge subset)
- Science reasoning questions
- Multiple choice with explanations

### Format for GRPO

```python
def format_for_grpo(example):
    # GRPO expects problem statements
    # Model will generate solutions
    return {
        "problem": example['question'],
        # No ground truth needed - model explores solutions
    }
```

## Colab 5: Continued Pretraining (Language Learning)

### Required Format
For teaching a new language, you need:
1. Monolingual text in target language
2. Parallel translation data (optional, helpful)
3. Large quantity (100k+ examples)

### Recommended Datasets

#### 1. Wikipedia Dumps
**Dataset**: `wikipedia` (any language)
```python
dataset = load_dataset("wikipedia", "20220301.es")  # Spanish
dataset = load_dataset("wikipedia", "20220301.fr")  # French
dataset = load_dataset("wikipedia", "20220301.de")  # German
```

#### 2. MC4 (Multilingual C4)
**Dataset**: `mc4`
- 101 languages
- Web-crawled text
- Cleaned and filtered

```python
dataset = load_dataset("mc4", "es")  # Spanish subset
```

#### 3. OSCAR
**Dataset**: `oscar-corpus/OSCAR-2301`
- 151 languages
- Deduplicated web text
- Multiple size variants

#### 4. Parallel Translation Data
**Dataset**: `wmt19` or `wmt21`
- English ↔ [language] pairs
- High-quality translations
- Good for teaching language mapping

```python
dataset = load_dataset("wmt19", "de-en")  # German-English
```

### Low-Resource Languages

For languages with limited data:

**Dataset**: `CohereForAI/aya-dataset`
- 101 languages including low-resource
- Instruction data
- Community-contributed

**Dataset**: `Davlan/masakhaner`
- African languages NER
- 10 languages

## Dataset Preparation Tips

### 1. Size Considerations

| Model Size | Min Examples | Recommended | Max Effective |
|-----------|--------------|-------------|---------------|
| SmolLM (135M) | 1,000 | 10,000 | 50,000 |
| Gemma-1B | 5,000 | 25,000 | 100,000 |
| Llama-3-8B | 10,000 | 50,000 | 500,000 |
| Mistral-7B | 10,000 | 50,000 | 500,000 |

**Rule of thumb**: More data = better, but quality > quantity

### 2. Data Quality Checklist

- [ ] No duplicates
- [ ] Consistent formatting
- [ ] Appropriate length (not too short/long)
- [ ] Diverse examples
- [ ] Clean text (no encoding issues)
- [ ] Balanced topics (if applicable)

### 3. Preprocessing Pipeline

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("yahma/alpaca-cleaned")

# 1. Filter by length
dataset = dataset.filter(lambda x: len(x['output']) > 20)

# 2. Remove duplicates
dataset = dataset.unique('instruction')

# 3. Shuffle
dataset = dataset.shuffle(seed=42)

# 4. Split train/validation
dataset = dataset.train_test_split(test_size=0.1)

# 5. Apply chat template
def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_example)
```

### 4. Custom Dataset Creation

```python
# From CSV
from datasets import Dataset
import pandas as pd

df = pd.read_csv("my_data.csv")
dataset = Dataset.from_pandas(df)

# From JSON
import json
with open("my_data.json", "r") as f:
    data = json.load(f)
dataset = Dataset.from_list(data)

# From scratch
data = [
    {"instruction": "...", "output": "..."},
    # ... more examples
]
dataset = Dataset.from_list(data)
```

### 5. Upload to HuggingFace

```python
# Login
from huggingface_hub import login
login()

# Push dataset
dataset.push_to_hub("your_username/dataset_name")

# Use later
dataset = load_dataset("your_username/dataset_name")
```

## Chat Templates by Model

Different models require different formatting:

### Llama 3.1
```python
template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
```

### Mistral
```python
template = "<s>[INST] {instruction} [/INST] {output}</s>"
```

### Gemma 2
```python
template = "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
```

### Phi-3
```python
template = "<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"
```

### SmolLM2 / TinyLlama (Alpaca Style)
```python
template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
```

## Dataset Statistics to Track

When preparing datasets, track these metrics:

```python
import numpy as np

def dataset_stats(dataset):
    lengths = [len(x['text'].split()) for x in dataset]

    print(f"Total examples: {len(dataset)}")
    print(f"Avg length: {np.mean(lengths):.0f} words")
    print(f"Median length: {np.median(lengths):.0f} words")
    print(f"Min length: {np.min(lengths)} words")
    print(f"Max length: {np.max(lengths)} words")
    print(f"Std dev: {np.std(lengths):.0f} words")

dataset_stats(dataset['train'])
```

## Useful Dataset Resources

- **HuggingFace Datasets Hub**: https://huggingface.co/datasets
- **OpenAI Dataset Collection**: https://github.com/openai/openai-cookbook
- **Awesome LLM Datasets**: https://github.com/Hannibal046/Awesome-LLM#datasets
- **Dataset Search**: https://datasetsearch.research.google.com/

## License Considerations

Always check dataset licenses:
- **MIT/Apache 2.0**: Can use commercially
- **CC-BY**: Attribution required
- **CC-BY-NC**: Non-commercial only
- **CC-BY-SA**: Share-alike (derivatives must use same license)

Popular datasets are usually permissive, but always verify!
