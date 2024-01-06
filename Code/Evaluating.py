# Define the evaluate function
def evaluate(dataloader):
    model.eval()
    predictions, true_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
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

    return accuracy, precision, recall, f1

# Evaluate the model on the test dataset
test_accuracy, test_precision, test_recall, test_f1 = evaluate(test_dataloader)
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test F1 Score: {test_f1}")
