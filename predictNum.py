import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Function to create sequences for training
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Define model parameters
input_size = 4
hidden_size1 = 64
hidden_size2 = 32
output_size = 1

# Instantiate the model
model = SimpleNN(input_size, hidden_size1, hidden_size2, output_size)

# Create dataset
data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
seq_len = 4

# Create sequences
X, y = create_sequences(data, seq_len)

# Normalize data (if needed)
scaler = StandardScaler()
X = torch.tensor(scaler.fit_transform(X), dtype=torch.float32)

# Create DataLoader
batch_size = 2
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
print_every = 100

for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in data_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % print_every == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Testing the model
model.eval()
test_input = torch.tensor([[2, 4, 6, 8]], dtype=torch.float32)
with torch.no_grad():
    prediction = model(test_input)
    print(f'Predicted next number: {prediction.item():.4f}')