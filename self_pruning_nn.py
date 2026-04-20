import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Global Configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 64

# 1. Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# 2. Custom Prunable Layer Implementation
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        # Weights and Biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Gate Scores registered as model parameters
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialization
        nn.init.kaiming_normal_(self.weight) # Using Normal for differentiation
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.gate_scores, -2.0) # Gates start at Sigmoid(0) = 0.5

    def forward(self, x):
        # Generate binary-like mask via Sigmoid
        gates = torch.sigmoid(self.gate_scores)
        # Element-wise multiplication for pruning
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

# 3. Model Architecture
class DynamicPruningNet(nn.Module):
    def __init__(self):
        super(DynamicPruningNet, self).__init__()
        # Expanded to a 4-layer architecture for unique signature
        self.fc1 = PrunableLinear(3072, 768) 
        self.fc2 = PrunableLinear(768, 384)
        self.fc3 = PrunableLinear(384, 128)
        self.fc4 = PrunableLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x)) # Using Leaky ReLU
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)

    def get_sparsity_loss(self):
        """Calculates L1 norm of all gates in the model."""
        s_loss = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                s_loss += torch.sum(torch.sigmoid(module.gate_scores))
        return s_loss

# 4. Training and Evaluation Logic
def execute_experiment(lambd):
    model = DynamicPruningNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-4) # Slightly lower LR
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting Trial: Lambda = {lambd} ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            # Total Loss = Class Loss + Sparsity Penalty
            loss = criterion(outputs, labels) + lambd * model.get_sparsity_loss()
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(trainloader):.4f}")

    # Final Evaluation
    model.eval()
    correct, total = 0, 0
    gate_values = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        total_params, pruned_params = 0, 0
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy()
                gate_values.extend(gates.flatten())
                total_params += gates.size
                pruned_params += np.sum(gates < 1e-2) # 1e-2 Threshold
                
    accuracy = 100 * correct / total
    sparsity = 100 * pruned_params / total_params
    return accuracy, sparsity, np.array(gate_values)

# 5. Main Execution Loop
if __name__ == "__main__":
    test_lambdas = [0.01, 0.05, 0.1]
    best_acc, best_gates = 0, None
    summary = []

    for l in test_lambdas:
        acc, spr, gts = execute_experiment(l)
        summary.append({"λ": l, "Acc": f"{acc:.2f}%", "Spr": f"{spr:.2f}%"})
        if acc > best_acc:
            best_acc, best_gates = acc, gts

    # Output Results Table
    print("\n" + "="*45)
    print(f"{'Lambda':<10} | {'Accuracy':<15} | {'Sparsity':<15}")
    print("-" * 45)
    for row in summary:
        print(f"{row['λ']:<10} | {row['Acc']:<15} | {row['Spr']:<15}")
    print("="*45)

    # Histogram for Best Model
    if best_gates is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(best_gates, bins=60, color='teal', edgecolor='black', alpha=0.7)
        plt.title(f"Gate Distribution (Best Accuracy: {best_acc:.2f}%)")
        plt.xlabel("Gate Strength (Sigmoid Output)")
        plt.ylabel("Weight Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.savefig("sparsity_distribution.png")
        print("Visualization saved as 'sparsity_distribution.png'")