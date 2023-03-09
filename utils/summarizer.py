
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

model_name = "vblagoje/bart_lfqa"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# https://huggingface.co/vblagoje/bart_lfqa

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.to(device)

def summarize(question, context):
    
    query_and_docs = "question: {} context: {}".format(question, context)
    model_input = tokenizer(query_and_docs, truncation=True, padding=True, return_tensors="pt")
    
    generated_answers_encoded = model.generate(input_ids=model_input["input_ids"].to(device),
                                            attention_mask=model_input["attention_mask"].to(device),
                                            min_length=64,
                                            max_length=256,
                                            do_sample=False, 
                                            early_stopping=True,
                                            num_beams=8,
                                            temperature=1.0,
                                            top_k=None,
                                            top_p=None,
                                            eos_token_id=tokenizer.eos_token_id,
                                            no_repeat_ngram_size=3,
                                            num_return_sequences=1)
    return tokenizer.batch_decode(generated_answers_encoded, skip_special_tokens=True,clean_up_tokenization_spaces=True)
