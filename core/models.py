import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
from tqdm import tqdm

class SVMModel:
    def __init__(self, **kwargs):
        # Default high-performance configuration
        if 'probability' not in kwargs:
            kwargs['probability'] = True
        self.model = SVC(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, raw=False):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim, dropout_prob=0.3):
        super(ANN, self).__init__()
        # Three hidden layers: Input -> 128 -> 64 -> 32 -> Output
        self.layer1 = nn.Linear(input_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.layer2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.layer3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.bn3 = nn.BatchNorm1d(hidden_dim_3)
        self.relu3 = nn.ReLU()
        
        self.layer4 = nn.Linear(hidden_dim_3, output_dim)
        
        # Random parameter initialization (Kaiming Normal)
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.layer4.weight)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.layer1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.layer2(x))))
        x = self.relu3(self.bn3(self.layer3(x)))
        x = self.layer4(x)
        return x

class ANNModel:
    def __init__(self, input_dim, output_dim, lr=0.001, epochs=100, batch_size=64, 
                 dropout_prob=0.3, threshold=0.4, patience=10, device=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.threshold = threshold
        self.patience = patience
        
        # Device auto-detection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = ANN(
            input_dim=input_dim, 
            hidden_dim_1=128, 
            hidden_dim_2=64, 
            hidden_dim_3=32, 
            output_dim=output_dim, 
            dropout_prob=dropout_prob
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.loss_history = {'train': [], 'val': []}
        self.best_model_state = None

    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        self.loss_history = {'train': [], 'val': []}
        
        # Prepare training data loader
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Prepare validation data loader
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # Use tqdm progress bar if verbose
        epoch_range = range(self.epochs)
        if verbose:
            epoch_range = tqdm(epoch_range, desc="Training ANN Model")
            
        for epoch in epoch_range:
            # Training phase
            self.model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(X_train)
            self.loss_history['train'].append(train_loss)
            
            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            if val_loader is not None:
                self.model.eval()
                correct = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, dim=1)
                        correct += torch.sum(preds == targets).item()
                val_loss /= len(X_val)
                val_acc = correct / len(X_val)
                self.loss_history['val'].append(val_loss)
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(train_loss)
                
            # Log epoch metrics
            if verbose:
                if val_loader is not None:
                    epoch_range.set_postfix({
                        'Train Loss': f'{train_loss:.4f}', 
                        'Val Loss': f'{val_loss:.4f}', 
                        'Val Acc': f'{val_acc:.4f}'
                    })
                else:
                    epoch_range.set_postfix({'Train Loss': f'{train_loss:.4f}'})
                    
            # Early stopping check
            current_loss = val_loss if val_loader is not None else train_loss
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if verbose:
                        print(f"\n[Early Stopping] Triggered early stopping at epoch {epoch+1}. Restoring best model weights...")
                    break
                    
        # Restore best model state
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def predict(self, X, raw=False):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probabilities, dim=1)
            
            preds = preds.cpu().numpy()
            max_probs = max_probs.cpu().numpy()
            
            # Post-processing Softmax confidence threshold filtering
            if not raw:
                preds[max_probs < self.threshold] = -1
            
        return preds

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

class RFModel:
    def __init__(self, **kwargs):
        if 'n_jobs' not in kwargs:
            kwargs['n_jobs'] = -1  # Enable multi-core processing
        self.model = RandomForestClassifier(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, raw=False):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

if __name__ == "__main__":
    print("Generating dummy data...")
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 3, 100)
    X_val = np.random.rand(20, 10)
    y_val = np.random.randint(0, 3, 20)
    X_test = np.random.rand(20, 10)
    
    print("\nTesting SVMModel...")
    svm = SVMModel(C=2.0)
    svm.train(X_train, y_train)
    print("SVM Predictions:", svm.predict(X_test))
    
    print("\nTesting RFModel...")
    rf = RFModel(n_estimators=10)
    rf.train(X_train, y_train)
    print("RF Predictions:", rf.predict(X_test))
    
    print("\nTesting ANNModel...")
    ann = ANNModel(input_dim=10, output_dim=3, epochs=15, batch_size=8, patience=5)
    ann.train(X_train, y_train, X_val, y_val, verbose=True)
    print("ANN Predictions (-1 means rejected):", ann.predict(X_test))
    print("ANN Probabilities shape:", ann.predict_proba(X_test).shape)
    
    print("\nAll core models successfully initialized and tested.")
