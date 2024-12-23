def predict(model, tokenizer, sentence, aspect, device):
    t1 = tokenizer.tokenize(sentence)
    t2 = tokenizer.tokenize(aspect)

    word_pieces = ['[CLS]'] + t1 + ['[SEP]'] + t2

    segment_tensor = [0] + [0]*len(t1) + [0] + [1]*len(t2)

    input_ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([input_ids]).to(device)
    segment_tensor = torch.tensor([segment_tensor]).to(device)

    with torch.no_grad():
        outputs = model(input_tensor, None, segment_tensor, None)
        _, predictions = torch.max(outputs, dim=1)

    return word_pieces, int(predictions), outputs