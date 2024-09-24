def clean_suggestion(suggestion, tokenizer):
    if suggestion[-1] in tokenizer.all_special_tokens:
        suggestion = suggestion[:-1]
    last_word_boundary = suggestion.rfind(' ')
    if last_word_boundary != -1:
        suggestion = suggestion[:last_word_boundary]
    return suggestion

def suggest_next_words(prefix, model, tokenizer, device, max_length=60, strategy="greedy", num_beams=6, no_repeat_ngram_size=2):
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
    if strategy == "greedy":
        output = model.generate(input_ids=input_ids, max_length=max_length)
    else:
        raise ValueError("Invalid strategy")
    suggestions = [clean_suggestion(tokenizer.decode(seq, skip_special_tokens=True), tokenizer) for seq in output]
    beam_output = model.generate(input_ids=input_ids, max_length=max_length, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size)
    beam_suggestion = clean_suggestion(tokenizer.decode(beam_output[0], skip_special_tokens=True), tokenizer)
    return suggestions[0], beam_suggestion
