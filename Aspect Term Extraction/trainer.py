def train_epoch(model, optimizer, train_loader, device):
    losses = []
    for batch in (train_loader):
        ids_tensors, tags_tensors, _, masks_tensors = batch
        ids_tensors = ids_tensors.to(device)
        tags_tensors = tags_tensors.to(device)
        masks_tensors = masks_tensors.to(device)

        loss, _ = model(
            ids_tensors=ids_tensors,
            masks_tensors=masks_tensors,
            tags_tensors=tags_tensors
        )
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return sum(losses)/len(losses)

def evaluate_epoch(model, valid_loader, device):
    losses = []

    preds, labels = [], []
    with torch.no_grad():
        for batch in (valid_loader):
            ids_tensors, tags_tensors, _, masks_tensors = batch
            ids_tensors = ids_tensors.to(device)
            tags_tensors = tags_tensors.to(device)
            masks_tensors = masks_tensors.to(device)

            loss, outputs = model(
                ids_tensors=ids_tensors,
                masks_tensors=masks_tensors,
                tags_tensors=tags_tensors
            )
            losses.append(loss.item())

            _, p = torch.max(outputs, dim=2)
            preds += list([int(j) for i in p for j in i ])
            labels += list([int(j) for i in tags_tensors for j in i ])

    acc = np.mean(np.array(preds) == np.array(labels))
    return sum(losses)/len(losses), acc

def train(model, model_name, save_model, optimizer, train_loader, valid_loader, num_epochs, device):
    train_losses = []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100
    times = []
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        # Training
        train_loss = train_epoch(model, optimizer, train_loader, device)
        train_losses.append(train_loss)

        # Evaluation
        eval_loss, eval_acc = evaluate_epoch(model, valid_loader, device)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        # Save best model
        if eval_loss < best_loss_eval:
            torch.save(model.state_dict(), save_model + f'/{model_name}.pt')

        times.append(time.time() - epoch_start_time)
        # Print loss, acc end epoch
        print("-" * 59)
        print(
            "| End of epoch {:3d} | Time: {:5.2f}s | Train Loss {:8.3f} "
            "| Valid Accuracy {:8.3f} | Valid Loss {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, train_loss, eval_acc, eval_loss
            )
        )
        print("-" * 59)

    # Load best model
    model.load_state_dict(torch.load(save_model + f'/{model_name}.pt'))
    model.eval()
    metrics = {
        'train_loss': train_losses,
        'valid_accuracy': eval_accs,
        'valid_loss': eval_losses,
        'time': times
    }
    return model, metrics