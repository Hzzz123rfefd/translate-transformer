from transformers import MBartForConditionalGeneration, MBartTokenizer
model_name_or_path = 'facebook/mbart-large-50-many-to-many-mmt'
tokenizer = MBartTokenizer.from_pretrained(model_name_or_path)
model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)