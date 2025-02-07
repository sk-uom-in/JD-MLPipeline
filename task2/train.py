from LSTMClassifier import * 
from Imports import *

def train_model(model, train_loader, save_path, epochs=50):
    model.train()
    best_accuracy = 0           

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move to GPU

            optimizer.zero_grad()  # Reset gradients
            
            with torch.autocast(device_type="cuda"):  # Mixed Precision Training
                output = model(X_batch)
                # print("x: ",X_batch[:1])
                # print("y: ",output[:2])
                loss = loss_function(output, y_batch) 
                # print("loss: ", loss)

            scaler.scale(loss).backward()  # Backpropagation with scaling
            scaler.step(optimizer)  # Optimizer step
            scaler.update()  # Update scaler

            total_loss += loss.item()
            predictions = torch.argmax(output, dim=1)
            # print("predictions: ", predictions[:1])
            # print("y_real: ",y_batch[:1])
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Compute epoch loss & accuracy
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Save model if accuracy improves
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with Accuracy: {epoch_accuracy:.4f}")

# Train the model
train_model(model, train_loader, save_path="last_lstm_50.pth", epochs=50)







def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move to GPU
            output = model(X_batch)
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    print(f"Test Accuracy: {correct/total:.4f}")

# Evaluate the model
evaluate_model(model, test_loader)
