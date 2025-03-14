def objective(trial):
  # Generate the model.
  model = ConvNet(trial).to(DEVICE)
  # Generate the optimizers.
  # try RMSprop and SGD
  '''
  optimizer_name = trial.suggest_categorical("optimizer", ["RMSprop", "SGD"])
  momentum = trial.suggest_float("momentum", 0.0, 1.0)
  lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
  optimizer = getattr(optim, optimizer_name)
  (model.parameters(), lr=lr,momentum=momentum)
  '''
  #try Adam, AdaDelta adn Adagrad
  optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adadelta","Adagrad"])
  lr = trial.suggest_float("lr", 1e-5, 1e-1,log=True)
  optimizer = getattr(optim, optimizer_name)
  (model.parameters(), lr=lr)
  batch_size=trial.suggest_int("batch_size", 64, 256,step=64)
  criterion=nn.CrossEntropyLoss()
  # Get the MNIST imagesset.
  train_loader, valid_loader = get_mnist(train_dataset,batch_size)
  # Training of the model.
  for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
      # Limiting training images for faster epochs.
      #if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
      #    break
      images, labels = images.to(DEVICE), labels.to(DEVICE)
      optimizer.zero_grad()
      output = model(images)
      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()
      # Validation of the model.
      model.eval()
      correct = 0
      with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(valid_loader):
          # Limiting validation images.
          # if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
          #    break
          images, labels = images.to(DEVICE), labels.to(DEVICE)
          output = model(images)
          # Get the index of the max log-probability.
          pred = output.argmax(dim=1, keepdim=True)
          correct += pred.eq(labels.view_as(pred)).sum().item()
          accuracy = correct / len(valid_loader.dataset)
          trial.report(accuracy, epoch)
          # Handle pruning based on the intermediate value.
          if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            return accuracy