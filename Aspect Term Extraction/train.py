save_model = "./model"
model = ABTEBert(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 5
best_model, metrics = train(
    model, model_name, save_model, optimizer, train_loader, test_loader, num_epochs, device
)