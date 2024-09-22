# llama_model.py

"""
Language Model Module

This module contains functions to initialize, fine-tune, and utilize a language model
to interpret LSTM outputs, analyze textual data, generate reports, and handle user queries.

Functions:
- initialize_language_model()
- fine_tune_language_model(language_model, tokenizer, domain_data)
- interpret_lstm_output(language_model, tokenizer, lstm_predictions, context_data)
- analyze_textual_data(language_model, tokenizer, textual_data)
- generate_report(language_model, tokenizer, interpretation_results, insights)
- respond_to_query(language_model, tokenizer, query, context_data)
"""

from transformers import LLaMAForCausalLM, LLaMATokenizer
import torch
from main import DEVICE
def initialize_language_model(): 
    """
    Initialize the LLaMA-3 language model and tokenizer.

    Returns:
    - language_model: The initialized LLaMA-3 language model.
    - tokenizer: The tokenizer associated with the language model.
    """
    try:
        # Initialize the LLaMA model and tokenizer
        language_model = LLaMAForCausalLM.from_pretrained("llama-3")
        tokenizer = LLaMATokenizer.from_pretrained("llama-3")
        language_model.eval()   # Set model to evaluation mode
        print("Language model initialized.")
        return language_model, tokenizer
    except Exception as e:
        print(f"Error in initializing language model: {e}")
        return None, None

def fine_tune_language_model(language_model, tokenizer, domain_data, epochs=1, batch_size=8, learning_rate=5e-5):
    """
    Fine-tune the language model on domain-specific data.

    Parameters:
    - language_model: The initialized language model.
    - tokenizer: The tokenizer associated with the language model.
    - domain_data: A list of domain-specific textual data.
    - epochs (int): Number of epochs to fine-tune.
    - batch_size (int): Batch size for fine-tuning.
    - learning_rate (float): Learning rate for the optimizer.

    Returns:
    - fine_tuned_model: The fine-tuned language model.
    """
    try:
        from torch.utils.data import DataLoader, Dataset
        from transformers import AdamW
        from tqdm import tqdm

        class DomainDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=512):
                self.input_ids = []
                self.attention_masks = []
                for text in texts:
                    encodings_dict = tokenizer(
                        text,
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                        return_tensors='pt'
                    )
                    self.input_ids.append(encodings_dict['input_ids'][0])
                    self.attention_masks.append(encodings_dict['attention_mask'][0])

            def __len__(self):
                return len(self.input_ids)

            def __getitem__(self, idx):
                return {
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_masks[idx]
                }

        # Prepare the dataset
        dataset = DomainDataset(domain_data, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set up the optimizer
        optimizer = AdamW(language_model.parameters(), lr=learning_rate)

        device = DEVICE
        language_model.to(device)

        language_model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0
            for batch in tqdm(dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = language_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            print(f"Average Loss: {avg_loss}")

        print("Language model fine-tuned on domain-specific data.")
        return language_model
    except Exception as e:
        print(f"Error in fine-tuning language model: {e}")
        return None

def interpret_lstm_output(language_model, tokenizer, lstm_predictions, context_data):
    """
    Use the language model to interpret LSTM outputs and provide context.

    Parameters:
    - language_model: The fine-tuned language model.
    - tokenizer: The tokenizer associated with the language model.
    - lstm_predictions: The predictions from the LSTM model.
    - context_data: Additional context data (e.g., operational parameters).

    Returns:
    - interpretation_results: A textual interpretation of the LSTM outputs.
    """
    try:
        # Prepare input prompt for the language model
        prompt = generate_interpretation_prompt(lstm_predictions, context_data)

        # Generate interpretation using the language model
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        device = DEVICE
        language_model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Generate output
        output_ids = language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=500,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        interpretation = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return interpretation
    except Exception as e:
        print(f"Error in interpreting LSTM outputs: {e}")
        return None

def analyze_textual_data(language_model, tokenizer, textual_data):
    """
    Analyze maintenance logs and operator notes for relevant insights.

    Parameters:
    - language_model: The fine-tuned language model.
    - tokenizer: The tokenizer associated with the language model.
    - textual_data: A DataFrame containing textual data.

    Returns:
    - insights: Extracted insights from the textual data.
    """
    try:
        # Combine textual data into a single string
        combined_text = ' '.join(textual_data['text'].tolist())

        # Prepare prompt
        prompt = f"Analyze the following maintenance logs and operator notes for relevant insights:\n{combined_text}"

        # Generate insights using the language model
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        device = DEVICE
        language_model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Generate output
        output_ids = language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=500,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        insights = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return insights
    except Exception as e:
        print(f"Error in analyzing textual data: {e}")
        return None

def generate_report(language_model, tokenizer, interpretation_results, insights):
    """
    Generate a comprehensive report combining interpretations and insights.

    Parameters:
    - language_model: The fine-tuned language model.
    - tokenizer: The tokenizer associated with the language model.
    - interpretation_results: The interpretation of LSTM outputs.
    - insights: Insights extracted from textual data.

    Returns:
    - report: A human-readable report.
    """
    try:
        # Prepare input prompt for report generation
        prompt = generate_report_prompt(interpretation_results, insights)

        # Generate report using the language model
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        device = DEVICE
        language_model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Generate output
        output_ids = language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1000,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        report = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return report
    except Exception as e:
        print(f"Error in generating report: {e}")
        return None

def respond_to_query(language_model, tokenizer, query, context_data):
    """
    Respond to natural language queries about motor status and predictions.

    Parameters:
    - language_model: The fine-tuned language model.
    - tokenizer: The tokenizer associated with the language model.
    - query: The user's natural language query.
    - context_data: Additional context data for generating accurate responses.

    Returns:
    - response: The language model's response to the query.
    """
    try:
        # Prepare input prompt including the query and context
        prompt = generate_query_prompt(query, context_data)

        # Generate response using the language model
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        device = DEVICE
        language_model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Generate output
        output_ids = language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=500,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return response
    except Exception as e:
        print(f"Error in responding to query: {e}")
        return None

# Helper functions for prompt generation
def generate_interpretation_prompt(lstm_predictions, context_data):
    """
    Generate a prompt for the language model to interpret LSTM outputs.

    Parameters:
    - lstm_predictions: The predictions from the LSTM model.
    - context_data: Additional context data.

    Returns:
    - prompt: A string prompt for the language model.
    """
    prompt = f"Interpret the following LSTM predictions:\n{lstm_predictions}\n\nContext:\n{context_data}"
    return prompt

def generate_report_prompt(interpretation_results, insights):
    """
    Generate a prompt for the language model to create a comprehensive report.

    Parameters:
    - interpretation_results: The interpretation of LSTM outputs.
    - insights: Insights from textual data.

    Returns:
    - prompt: A string prompt for the language model.
    """
    prompt = f"Generate a comprehensive report based on the following information:\n\nInterpretation:\n{interpretation_results}\n\nInsights:\n{insights}"
    return prompt

def generate_query_prompt(query, context_data):
    """
    Generate a prompt for the language model to respond to a user query.

    Parameters:
    - query: The user's natural language query.
    - context_data: Additional context data.

    Returns:
    - prompt: A string prompt for the language model.
    """
    prompt = f"User query: {query}\n\nContext:\n{context_data}\n\nProvide a detailed and helpful response."
    return prompt
