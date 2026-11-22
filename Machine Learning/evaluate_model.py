import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

# 1. Hardware Setup (Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running evaluation on device: {device}\n")

# 2. Load the Dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'insurance_ready_for_training.csv')
df = pd.read_csv(csv_path)

# Separate features (X) and actual charges (y)
X = df.drop('charges', axis=1).values
y_actual = df['charges'].values

# Calculate the mean and std of the actual charges to un-scale predictions later
y_mean = y_actual.mean()
y_std = y_actual.std()

# Convert inputs to PyTorch tensors
X_tensor = torch.FloatTensor(X).to(device)

# 3. Define the exact same Architecture
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

# 4. Load the Saved Model Weights
model = InsurancePredictor().to(device)
model_path = os.path.join(script_dir, 'insurance_model.pth')

# Load the weights into the architecture
model.load_state_dict(torch.load(model_path, weights_only=True))

# VERY IMPORTANT: Put the model in evaluation mode (turns off training behaviors)
model.eval()

print("Model loaded successfully. Generating predictions...\n")

# 5. Make Predictions on the Entire Dataset
with torch.no_grad():
    # Pass all inputs through the model at once
    predictions_scaled = model(X_tensor)
    
    # Move predictions back to the CPU, convert to numpy, and flatten the array
    predictions_scaled_np = predictions_scaled.cpu().numpy().flatten()
    
    # Un-scale to get real dollar amounts
    predictions_dollars = (predictions_scaled_np * y_std) + y_mean

# 6. Compare and Analyze the Results
# Create a DataFrame to easily view the side-by-side comparison
results_df = pd.DataFrame({
    'Actual_Charge': y_actual,
    'Predicted_Charge': predictions_dollars
})

# Calculate how far off each prediction was
results_df['Difference'] = np.abs(results_df['Actual_Charge'] - results_df['Predicted_Charge'])

# Print a formatted side-by-side comparison of the first 15 patients
print("--- SIDE-BY-SIDE COMPARISON (First 15 Patients) ---")
for index, row in results_df.head(15).iterrows():
    actual = row['Actual_Charge']
    predicted = row['Predicted_Charge']
    diff = row['Difference']
    print(f"Patient {index:02d} | Actual: ${actual:9.2f} | Predicted: ${predicted:9.2f} | Off by: ${diff:7.2f}")

# Calculate and print the overall accuracy metric across all 1338 rows
average_error = results_df['Difference'].mean()
print(f"\n--- OVERALL PERFORMANCE ON TRAINING DATA ---")
print(f"Average Error across all {len(results_df)} patients: ${average_error:,.2f}")