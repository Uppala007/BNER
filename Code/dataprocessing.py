import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import DataLoader, TensorDataset

# Function to parse the dataset file
def parse_data(file_path):
    texts, tags = [], []
    current_text, current_tags = '', []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 1:  # New text starts
                    if current_text:
                        texts.append(current_text)
                        tags.append(current_tags)
                    current_text = parts[0].split('|')[2]  # Extract text
                    current_tags = ['O'] * len(current_text)  # Initialize tags
                else:
                    # Parse entity annotation
                    start, end = int(parts[1]), int(parts[2])
                    entity = parts[3]
                    tag = parts[4]
                    # Ensure the tag list is long enough
                    current_tags.extend(['O'] * (end - len(current_tags)))
                    for i in range(start, end):
                        prefix = 'B-' if i == start else 'I-'
                        current_tags[i] = prefix + tag
        # Add last text
        if current_text:
            texts.append(current_text)
            tags.append(current_tags)
    return texts, tags

# Function to tokenize and encode the dataset
def encode_dataset(tokenizer, texts, tags, max_length):
    input_ids = []
    attention_masks = []
    label_ids = []

    for text, text_tags in zip(texts, tags):
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        # Adjust the length of text_tags to match max_length
        adjusted_tags = text_tags[:max_length]
        padding_length = max_length - len(adjusted_tags)
        adjusted_tags += ['O'] * padding_length

        # Create a list of label IDs from the tag names
        label_id_list = [tag_to_id.get(tag, 0) for tag in adjusted_tags]

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        label_ids.append(torch.unsqueeze(torch.tensor(label_id_list), 0))

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    label_ids = torch.cat(label_ids, dim=0)

    return input_ids, attention_masks, label_ids

# Function to extract unique tags from datasets
def get_unique_tags(*datasets):
    unique_tags = set()
    for dataset in datasets:
        for tags in dataset:
            unique_tags.update(tags)
    return unique_tags

# Load datasets
train_texts, train_tags = parse_data('/content/drive/My Drive/Dataset/NCBItrainset_corpus.txt')
test_texts, test_tags = parse_data('/content/drive/My Drive/Dataset/NCBItestset_corpus.txt')
dev_texts, dev_tags = parse_data('/content/drive/My Drive/Dataset/NCBIdevelopset_corpus.txt')

# Extract unique tags from all datasets and create tag_to_id dictionary
unique_tags = get_unique_tags(train_tags, test_tags, dev_tags)
tag_to_id = {tag: id for id, tag in enumerate(unique_tags)}
tag_to_id['O'] = 0  # Ensure 'O' tag is always mapped to 0

# Initialize BioBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
model = AutoModelForTokenClassification.from_pretrained(
    'dmis-lab/biobert-v1.1',
    num_labels=len(tag_to_id),
    output_attentions=False,
    output_hidden_states=False
)

# Send model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Tokenize and encode datasets
train_input_ids, train_attention_masks, train_label_ids = encode_dataset(tokenizer, train_texts, train_tags, max_length=128)
test_input_ids, test_attention_masks, test_label_ids = encode_dataset(tokenizer, test_texts, test_tags, max_length=128)
dev_input_ids, dev_attention_masks, dev_label_ids = encode_dataset(tokenizer, dev_texts, dev_tags, max_length=128)

# Diagnostic code to check the lengths of tensors
print("Length of train_input_ids:", len(train_input_ids))
print("Length of train_attention_masks:", len(train_attention_masks))
print("Length of train_label_ids:", len(train_label_ids))

print("Length of test_input_ids:", len(test_input_ids))
print("Length of test_attention_masks:", len(test_attention_masks))
print("Length of test_label_ids:", len(test_label_ids))

print("Length of dev_input_ids:", len(dev_input_ids))
print("Length of dev_attention_masks:", len(dev_attention_masks))
print("Length of dev_label_ids:", len(dev_label_ids))

# Create TensorDatasets
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_label_ids)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_label_ids)
dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_label_ids)

# Create DataLoaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
