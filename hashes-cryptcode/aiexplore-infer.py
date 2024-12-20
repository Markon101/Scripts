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

# Load the model
model.load_state_dict(torch.load('char_rnn_model.pth'))
model.to(device)

# Function to convert string to tensor
def string_to_tensor(string):
    tensor = torch.zeros(len(string), 1, input_size)  # Shape: (sequence_length, batch_size, input_size)
    for i, char in enumerate(string):
        tensor[i][0][ord(char) % input_size] = 1.0
    return tensor.to(device)

def interactive_inference(model):
    model.eval()
    while True:
        input_string = input("Enter a string to predict the next character: ")
        if input_string.lower() == 'exit':
            break
        input_tensor = string_to_tensor(input_string)
        batch_size = input_tensor.size(1)  # Get the dynamic batch size
        hidden = model.init_hidden(batch_size)  # Adjust hidden state to match batch size
        with torch.no_grad():
            output, _ = model(input_tensor, hidden)
        predicted_index = torch.argmax(output[-1], dim=1).item()
        predicted_char = chr(predicted_index)
        print(f"The predicted next character is: {predicted_char}")

# Run the interactive inference loop
interactive_inference(model)

