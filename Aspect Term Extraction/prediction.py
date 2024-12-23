def predict(best_model, sentence, device):
    word_pieces = list(tokenizer.tokenize(sentence))
    input_ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        outputs = model(input_tensor, None, None)
        _, predictions = torch.max(outputs, dim=2)

    predictions = predictions[0].tolist()
    return word_pieces, predictions, outputs

sentence = " ".join(test_df.iloc[0]["Tokens"].replace("'", "").strip("][").split(', '))
predict(best_model, sentence, device)