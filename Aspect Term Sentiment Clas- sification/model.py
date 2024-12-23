class ABSABert(torch.nn.Module):
    def __init__(self, model_name):
        super(ABSABert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, masks_tensors, segments_tensors, lable_tensors):
        out_dict = self.bert(
            input_ids=ids_tensors,
            attention_mask=masks_tensors,
            token_type_ids=segments_tensors
        )
        linear_outputs = self.linear(out_dict['pooler_output'])

        if lable_tensors is not None:
            loss = self.loss_fn(linear_outputs, lable_tensors)
            return loss, linear_outputs
        else:
            return linear_outputs