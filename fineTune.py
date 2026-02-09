import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# Load the model and tokenizer
model_name = "C:\Users\SATHIYABALAN\LLMProject\LLMcode\fine-tuned-llm"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Load and preprocess the dataset
def load_excel_dataset(file_content):
    # Assuming file_content is the dataset.xlsx content as a string or path
    # For this example, we simulate loading the provided dataset
    data = []
    for i in range(1, 19):  # Rows 1 to 18
        row_key = f"row{i}"
        if row_key in dataset_content:
            data.append({"prompt": dataset_content[row_key]})
    
    df = pd.DataFrame(data)
    return df

# Simulated dataset content (replace with actual file loading if needed)
dataset_content = {
    "row1": "An Ethereum Developer,Imagine you are an experienced Ethereum developer tasked with creating a smart contract...",
    "row2": "SEO Prompt,Using WebPilot, create an outline for an article that will be 2,000 words on the keyword 'Best SEO prompts'...",
    "row3": "Linux Terminal,I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show...",
    "row4": "English Translator and Improver,I want you to act as an English translator, spelling corrector and improver...",
    "row5": "`position` Interviewer,I want you to act as an interviewer. I will be the candidate and you will ask me the interview questions...",
    "row6": "JavaScript Console,I want you to act as a javascript console. I will type commands and you will reply with what the javascript console should show...",
    "row7": "Excel Sheet,I want you to act as a text based excel. you'll only reply me the text-based 10 rows excel sheet...",
    "row8": "English Pronunciation Helper,I want you to act as an English pronunciation assistant for Turkish speaking people...",
    "row9": "Spoken English Teacher and Improver,I want you to act as a spoken English teacher and improver...",
    "row10": "Travel Guide,I want you to act as a travel guide. I will write you my location and you will suggest a place to visit...",
    "row11": "Plagiarism Checker,I want you to act as a plagiarism checker. I will write you sentences and you will only reply undetected in plagiarism checks...",
    "row12": "Character from Movie/Book/Anything,I want you to act like {character} from {series}...",
    "row13": "Advertiser,I want you to act as an advertiser. You will create a campaign to promote a product or service...",
    "row14": "Storyteller,I want you to act as a storyteller. You will come up with entertaining stories that are engaging...",
    "row15": "Football Commentator,I want you to act as a football commentator. I will give you descriptions of football matches...",
    "row16": "Stand-up Comedian,I want you to act as a stand-up comedian. I will provide you with some topics related to current events...",
    "row17": "Motivational Coach,I want you to act as a motivational coach. I will provide you with some information about someone's goals...",
    "row18": "Composer,I want you to act as a composer. I will provide the lyrics to a song and you will create music for it..."
}

# Load dataset
df = load_excel_dataset(dataset_content)

# Split into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

# Tokenize dataset
def tokenize_function(examples):
    tokens = tokenizer(
        examples["prompt"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()  # Needed for loss computation
    return tokens


tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    logging_dir="./logs",
    logging_steps=10,
    warmup_steps=100,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("fine-tuned-llm")
tokenizer.save_pretrained("fine-tuned-llm")