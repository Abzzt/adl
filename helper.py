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
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            valid_loss += loss.item()
    
    valid_loss /= len(valid_loader)
    valid_accuracy = 100 * correct / total

    return valid_loss, valid_accuracy