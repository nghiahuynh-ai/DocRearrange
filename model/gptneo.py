from transformers import (
    GPTNeoForCausalLM, 
    GPT2Tokenizer,
)


model = GPTNeoForCausalLM.from_pretrained("NlpHUST/gpt-neo-vi-small")
tokenizer = GPT2Tokenizer.from_pretrained("NlpHUST/gpt-neo-vi-small")


print(model)
print(sum(p.numel() for p in model.parameters()))