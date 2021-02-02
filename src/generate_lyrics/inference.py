from .MXMGPT2LMHeadModel import adjust_length_to_model


def inference(prompt_text, tokenizer, model, phonemes_dict, length, temperature, top_p, top_k, repetition_penalty, do_sample, do_rhyming, stop_token, no_repeat_ngram_size):
    
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt
    
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=length + len(encoded_prompt[0]),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        do_rhyming=do_rhyming,
        num_return_sequences=1,
        phonemes_dict=phonemes_dict,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    
    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]
        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )
        generated_sequences.append(total_sequence)
    
    return generated_sequences[0]