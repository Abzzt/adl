import torch

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_train_loss = 0.0
    total = 0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        # outputs_binary = torch.where(outputs >= threshold, torch.tensor(1, device=outputs.device), torch.tensor(0, device=outputs.device))

        # print("outputs", outputs_binary)

        # total += labels.size(0)
        # correct += (outputs_binary == labels).sum().item()
        # correct = torch.sum(torch.eq(labels, outputs_binary).all(dim=1)).item()
        
        epoch_train_loss += loss.item()

    # train_accuracy = 100 * correct / total
        
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