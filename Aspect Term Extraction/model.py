class ABTEBert(torch.nn.Module):
    def __init__(self, model_name):
        super(ABTEBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, masks_tensors, tags_tensors):
        bert_outputs= self.bert(
            input_ids=ids_tensors, attention_mask=masks_tensors, return_dict=False
            )
        bert_outputs = bert_outputs[0]

        linear_outputs = self.linear(bert_outputs)
        if tags_tensors is not None:
            tags_tensors = tags_tensors.view(-1)
            linear_outputs_ = linear_outputs.view(-1,3)
            loss = self.loss_fn(linear_outputs_, tags_tensors)
            return loss, linear_outputs
        else:
            return linear_outputs