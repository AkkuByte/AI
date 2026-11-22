import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

# 1. Hardware Optimization for Apple Silicon (M4)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Initializing training on device: {device}\n")

# 2. Load the Prepared Data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'insurance_ready_for_training.csv')
df = pd.read_csv(csv_path)

# Separate the 8 input features and the 1 target variable
X = df.drop('charges', axis=1).values
y = df['charges'].values

# --- FIX 1: Scale the Target Variable (Charges) ---
# We standardize 'y' so it hovers around 0. This stops huge medical bills 
# from skewing the neural network's predictions.
y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std

# Convert the data into PyTorch Tensors and send them to the M4 chip
X_tensor = torch.FloatTensor(X).to(device)
# Notice we are passing the SCALED target to the tensor now
y_tensor = torch.FloatTensor(y_scaled).view(-1, 1).to(device) 

# 3. Define the Neural Network Architecture
class InsurancePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

model = InsurancePredictor().to(device)

# 4. Set the Loss Function and Optimizer
criterion = nn.L1Loss() 

# --- FIX 2: Lower Learning Rate ---
# Reduced from 0.05 to 0.01 for more stable learning
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. The Training Loop
# --- FIX 3: Increased Epochs ---
# Increased from 1500 to 4000 to allow for the smaller learning rate steps
epochs = 4000
print("Starting Neural Network Training...")

for epoch in range(epochs):
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress every 500 epochs
    if (epoch + 1) % 500 == 0:
        # Convert the scaled error back into real dollars for the printout
        loss_dollars = loss.item() * y_std
        print(f'Epoch {epoch+1}/{epochs} | Average Error: ${loss_dollars:.2f}')

print("\nTraining Complete!")

# 6. Quick Test: Compare a prediction against actual data
model.eval()
with torch.no_grad():
    sample_input = X_tensor[0]
    
    # Get the actual charge (from the original unscaled 'y' array)
    actual_charge = y[0] 
    
    # The model predicts a SCALED number. 
    # We must un-scale it to convert it back into real-world dollars!
    predicted_scaled = model(sample_input).item()
    predicted_charge = (predicted_scaled * y_std) + y_mean
    
    print("\n--- Real World Prediction Test ---")
    print(f"Actual Medical Charge:    ${actual_charge:,.2f}")
    print(f"Model's Predicted Charge: ${predicted_charge:,.2f}")

    # Save the trained model parameters
model_path = os.path.join(script_dir, 'insurance_model.pth')
torch.save(model.state_dict(), model_path)
print(f"\nModel successfully saved to: {model_path}")