import torch
import pandas as pd
import matplotlib.pyplot as plt

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

def test(model, test_loader, criterion, device, num_classes):
    # Evaluate the model on the test dataset
    model.eval()
    test_loss = 0
    threshold = 0.5
    total_label = {label: 0 for label in range(num_classes)}
    true_positives = {label: 0 for label in range(num_classes)}
    true_negatives = {label: 0 for label in range(num_classes)}
    false_positives = {label: 0 for label in range(num_classes)}
    false_negatives = {label: 0 for label in range(num_classes)}
    losses = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            losses.append(loss.item())
            outputs_binary = torch.where(outputs >= threshold, torch.tensor(1, device=outputs.device), torch.tensor(0, device=outputs.device))

            for label in range(num_classes):
                for output, true_label in zip(outputs_binary[:, label], labels[:, label]):
                    if output.item() == 1 and true_label.item() == 1:
                        true_positives[label] += 1
                    elif output.item() == 1 and true_label.item() == 0:
                        false_positives[label] += 1
                    elif output.item() == 0 and true_label.item() == 1:
                        false_negatives[label] += 1
                    else:
                        true_negatives[label] += 1
                        
                    total_label[label] += 1

    precision = {label: true_positives[label] / (true_positives[label] + false_positives[label]) if (true_positives[label] + false_positives[label]) != 0 else 0 for label in range(num_classes)}
    recall = {label: true_positives[label] / (true_positives[label] + false_negatives[label]) if (true_positives[label] + false_negatives[label]) != 0 else 0 for label in range(num_classes)}
    accuracy = {label: (true_positives[label] + true_negatives[label]) / total_label[label] if (true_positives[label] + true_negatives[label]) != 0 else 0 for label in range(num_classes)}
    f1 = {label: 2 * precision[label] * recall[label] / (precision[label] + recall[label]) if (precision[label] + recall[label]) != 0 else 0 for label in range(num_classes)}
    
    precision_df = pd.DataFrame.from_dict(precision, orient='index', columns=['Precision'])
    accuracy_df = pd.DataFrame.from_dict(accuracy, orient='index', columns=['Accuracy'])
    recall_df = pd.DataFrame.from_dict(recall, orient='index', columns=['Recall'])
    f1_df = pd.DataFrame.from_dict(f1, orient='index', columns=['F1-score'])

    metrics_df = pd.concat([precision_df, accuracy_df, recall_df, f1_df], axis=1)
    metrics_df.sort_index(inplace=True)


    return metrics_df, sum(losses) / len(losses)


def visualise_all_loss(train_loss_file, val_loss_file):
    # Load the training loss data
    train_loss_data = torch.load(train_loss_file)
    
    # Load the validation loss data
    val_loss_data = torch.load(val_loss_file)

    # Plot the loss
    for learning_rate, train_losses in train_loss_data.items():
        val_losses = val_loss_data[learning_rate]
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label=f"Train LR: {learning_rate}")
        plt.plot(epochs, val_losses, label=f"Validation LR: {learning_rate}", linestyle='--')

    # Add labels and legend
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs. Epoch for Different Learning Rates")
    plt.legend()
    plt.grid(True)
    plt.show()