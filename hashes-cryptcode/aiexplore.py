import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

# Hyperparameters
input_size = 128  # Size of the one-hot encoded input
hidden_size = 256 # You can adjust the size to control the number of parameters
output_size = 128 # Size of the one-hot encoded output
num_layers = 2    # Number of layers in the RNN

# Initialize the model and move it to the device (CPU or GPU)
model = CharRNN(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, file_path, chunk_size=200):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def __len__(self):
        # Implement a method to calculate the number of chunks
        return 1000000

    def __getitem__(self, idx):
        with open(self.file_path, 'r') as file:
            file.seek(idx * self.chunk_size)
            text = file.read(self.chunk_size)
        input_tensor = text_to_tensor(text[:-1])
        target_tensor = text_to_tensor(text[1:])
        return input_tensor, target_tensor, len(input_tensor)

# Function to convert text to tensor
def text_to_tensor(text):
    # Implement conversion from text to tensor
    return torch.tensor([ord(c) for c in text], dtype=torch.long)

# Collate function for DataLoader
def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    sequences, targets, lengths = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    targets_padded = pad_sequence(targets, batch_first=True)
    return sequences_padded, targets_padded, torch.tensor(lengths)

# Training method
def train(model, data_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch, (input, target, lengths) in enumerate(data_loader):
            input, target = input.to(device), target.to(device)
            lengths = lengths.to(device)
            hidden = model.init_hidden(input.size(0))
            input_packed = pack_padded_sequence(input, lengths, batch_first=True)
            output_packed, hidden = model(input_packed, hidden)
            output_padded, _ = pad_packed_sequence(output_packed, batch_first=True)
            loss = criterion(output_padded.transpose(1, 2), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Batch {batch}, Loss {loss.item()}')

# Inference method
def predict(model, input, hidden=None):
    model.eval()
    input = input.to(device)
    with torch.no_grad():
        output, hidden = model(input, hidden)
        return output, hidden

# Example usage
file_path = 'trainingdata.txt'
dataset = TextDataset(file_path)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Assuming `data_loader` is a PyTorch DataLoader object that provides batches of text data
# train(model, data_loader, criterion, optimizer)

# Save the model
torch.save(model.state_dict(), 'char_rnn_model.pth')

# Load the model and move it to the device
model.load_state_dict(torch.load('char_rnn_model.pth'))
model.to(device)

# Inference loop (predicting the next character)
# Assuming `input` is a tensor representing a sequence of characters
# output, hidden = predict(model, input)

