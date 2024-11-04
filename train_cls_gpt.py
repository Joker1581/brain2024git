import torch
import torch.nn as nn
from dataset import EEGDataset
from model.vit import ViT, SeqTransformer
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import os
import argparse
import wandb
import matplotlib.pyplot as plt

    
wandb.init(project="eeg_vit_vqvae", entity="ohicarip")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = './datasets'
anno_dir = os.path.join(data_dir, 'annotation_order.json')
eeg_dir = data_dir

def log_image_to_wandb(data, inout="input", train_or_test='train', epoch=0):
    """
    Function to plot the input data and log the image to WandB.
    
    Args:
    - data: numpy array of shape [ch, seq]
    - title: Title of the plot (optional)
    
    Returns:
    - None
    """
    seq, ch = data.shape
    
    # Plot the image
    fig = plt.figure(figsize=(15, 5))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel(f'Seq (Length: {seq})')
    plt.ylabel(f'Ch (Channels: {ch})')
    
    # Save the plot to a file buffer
    plt.savefig("temp_image.png")
    
    # Log the image to WandB
    wandb.log({f"epoch {train_or_test} {inout}": wandb.Image(fig)})
    
    # Close the plot to free resources
    plt.close()

def calc_vq_loss(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """
    rec_loss = nn.L1Loss()(pred, target)
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    quant_loss = quant_loss.mean()
    return quant_loss * quant_loss_weight + rec_loss, [rec_loss, quant_loss]

# Load the dataset
eeg_dataset = EEGDataset(eeg_dir, anno_dir, is_classification=True, is_all=False ,is_staring=True, is_imagination=False)
split_ratio = 0.8
train_size = int(split_ratio * len(eeg_dataset))
test_size = len(eeg_dataset) - train_size
indices = list(range(len(eeg_dataset)))
train_indices = indices[:train_size]  
test_indices = indices[train_size:]

train_dataset = Subset(eeg_dataset, train_indices)
test_dataset = Subset(eeg_dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define the model
model = ViT(patch_size=64, emb_dim=768, num_classes=8).to(device)

optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 20  # Define the number of epochs
for epoch in range(num_epochs):
    # Training Phase
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Compute loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct_predictions / total_predictions
    
    # Validation Phase
    model.eval()
    val_loss = 0.0
    correct_val_predictions = 0
    total_val_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Compute loss and accuracy
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_val_predictions += (predicted == labels).sum().item()
            total_val_predictions += labels.size(0)
    
    val_loss = val_loss / len(test_loader.dataset)
    val_accuracy = correct_val_predictions / total_val_predictions
    
    # Adjust learning rate
    scheduler.step()
    
    # Log metrics to WandB and console
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    })
    
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

