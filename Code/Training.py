import torch
import numpy as np
from tqdm import tqdm
from transformers import BertForTokenClassification, AutoTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize tokenizer and model for BioBERT
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')

# Cross-validation setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True)

# Iterate over each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f"Starting fold {fold+1}/{n_splits}")

    # Subset the datasets for the current fold
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    # Create DataLoaders for each subset
    train_dataloader = DataLoader(train_subset, batch_size=32)
    val_dataloader = DataLoader(val_subset, batch_size=32)

    # Initialize model for each fold
    model = BertForTokenClassification.from_pretrained(
        'dmis-lab/biobert-v1.1',
        num_labels=len(tag_to_id)
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            batch = tuple(b.to(device) for b in batch)
            b_input_ids, b_input_mask, b_labels = batch

            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Fold {fold+1}, Epoch {epoch+1}, Average train loss: {avg_train_loss}")

    # Validation
    model.eval()
    predictions, true_labels = [], []

    for batch in tqdm(val_dataloader, desc="Evaluating"):
        batch = tuple(b.to(device) for b in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    # Flatten lists and calculate metrics
    predictions = [p for sublist in predictions for p in sublist]
    true_labels = [l for sublist in true_labels for l in sublist]

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted', zero_division=0)
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"Fold {fold+1} - Val Accuracy: {accuracy}, Val Precision: {precision}, Val Recall: {recall}, Val F1: {f1}")
