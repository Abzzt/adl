import torch

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        
    return (epoch_train_loss / len(train_loader))

def evaluate(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            valid_loss += loss.item()
    
    valid_loss /= len(valid_loader)

    return valid_loss