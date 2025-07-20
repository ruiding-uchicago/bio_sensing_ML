#!/usr/bin/env python3
"""
Direct Regression for MLP with Split IgG Categories - Version 5

This script performs direct regression from signal data + experimental conditions 
to concentration using a Multi-Layer Perceptron (MLP) model.

Major Changes from No. 3:
1.  Focuses exclusively on the MLP model ('ConcentrationRegressor').
2.  Splits the 'IgG' target into 'IgG-Au40' and 'IgG-HRPAB' for all plots and metrics.
3.  Subplots are now a 2x2 grid for the four main categories.
4.  Hyperparameters are pre-set to the best-performing configuration from the grid search.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from collections import defaultdict
import argparse
warnings.filterwarnings('ignore')

# Optional imports for advanced boosting algorithms
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_data(data_dir):
    """Load all JSON files from the preprocessed directory."""
    data_list = []
    
    # Debug tracking
    nontarget_stats = {'total': 0, 'bacteria': 0, 'igg': 0, 'hrp': 0}
    concentration_stats = {'nontarget_not_minus19': [], 'nontarget_minus19': 0}
    
    # Get all JSON files
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {data_dir}")
    
    for filename in json_files:
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Check data structure
                if 'cycles' not in data or 'conditions' not in data:
                    print(f"Warning: {filename} missing required keys")
                    continue
                
                # Check cycles data shape
                cycles = np.array(data['cycles'])
                if cycles.shape != (200, 61, 3):
                    print(f"Warning: {filename} has unexpected cycle shape: {cycles.shape}")
                    # Pad or truncate if necessary
                    if len(cycles) < 200:
                        padding = np.zeros((200 - len(cycles), 61, 3))
                        cycles = np.vstack([cycles, padding])
                    elif len(cycles) > 200:
                        cycles = cycles[:200]
                    data['cycles'] = cycles.tolist()
                
                # Ensure Original_filename is set
                if 'Original_filename' not in data['conditions']:
                    data['conditions']['Original_filename'] = filename
                
                # Debug: Track IsNonTarget statistics
                conditions = data['conditions']
                if 'IsNonTarget' not in conditions:
                    print(f"DEBUG: {filename} missing IsNonTarget field - setting to False")
                    conditions['IsNonTarget'] = False
                
                is_nontarget = conditions['IsNonTarget']
                target = conditions.get('Target', 'Unknown')
                conc_transformed = conditions.get('Concentration_transformed', 0.0)
                
                if is_nontarget:
                    nontarget_stats['total'] += 1
                    if target == 'Bacteria':
                        nontarget_stats['bacteria'] += 1
                    elif target == 'IgG':
                        nontarget_stats['igg'] += 1
                    elif target == 'HRP':
                        nontarget_stats['hrp'] += 1
                    
                    if conc_transformed == -19.0:
                        concentration_stats['nontarget_minus19'] += 1
                    else:
                        concentration_stats['nontarget_not_minus19'].append({
                            'filename': filename,
                            'target': target,
                            'concentration_transformed': conc_transformed,
                            'is_nontarget': is_nontarget
                        })
                        print(f"DEBUG: NonTarget file with concentration != -19: {filename}, Target={target}, Conc_transformed={conc_transformed}")
                
                data_list.append(data)
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    print(f"Successfully loaded {len(data_list)} experiments from {data_dir}")
    
    # Print debug statistics
    print(f"\nDEBUG: NonTarget Statistics:")
    print(f"  Total NonTarget samples: {nontarget_stats['total']}")
    print(f"  Bacteria NonTarget: {nontarget_stats['bacteria']}")
    print(f"  IgG NonTarget: {nontarget_stats['igg']}")
    print(f"  HRP NonTarget: {nontarget_stats['hrp']}")
    print(f"  NonTarget with concentration = -19: {concentration_stats['nontarget_minus19']}")
    print(f"  NonTarget with concentration != -19: {len(concentration_stats['nontarget_not_minus19'])}")
    
    if concentration_stats['nontarget_not_minus19']:
        print("  Files with NonTarget=True but concentration != -19:")
        for item in concentration_stats['nontarget_not_minus19']:
            print(f"    {item}")
    
    # Print summary statistics
    targets = [d['conditions'].get('Target', 'Unknown') for d in data_list]
    setups = [d['conditions'].get('Setup', 'Unknown') for d in data_list]
    unique_targets = list(set(targets))
    unique_setups = list(set(setups))
    
    print(f"\nUnique targets: {unique_targets}")
    for target in unique_targets:
        count = targets.count(target)
        print(f"  {target}: {count} experiments")
    
    print(f"\nUnique setups: {unique_setups}")
    for setup in unique_setups:
        count = setups.count(setup)
        print(f"  {setup}: {count} experiments")
    
    return data_list

def create_category_based_split(data_list, random_seed=None):
    """
    Create train and hold-out sets where hold-out has one point from each category×concentration combination.
    
    Args:
        data_list: List of all data
        random_seed: Random seed for reproducible splits
        
    Returns:
        train_data: List of training data
        holdout_data: List of hold-out data
        holdout_info: Information about hold-out set composition
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Group data by category (setup + target) and concentration
    grouped_data = defaultdict(lambda: defaultdict(list))
    
    for idx, item in enumerate(data_list):
        conditions = item['conditions']
        setup = conditions.get('Setup', 'Unknown')
        target = conditions.get('Target', 'Unknown')
        concentration = conditions.get('Concentration_transformed', 0.0)
        
        # Create category key
        category = f"{setup}_{target}"
        
        # Round concentration to avoid floating point issues
        conc_key = round(concentration, 3)
        
        grouped_data[category][conc_key].append((idx, item))
    
    # Create hold-out set with one sample from each category×concentration
    train_indices = []
    holdout_indices = []
    holdout_info = []
    
    print("\nCreating category-based hold-out set:")
    for category, conc_dict in grouped_data.items():
        print(f"\nCategory: {category}")
        for concentration, samples in conc_dict.items():
            if len(samples) > 0:
                # Randomly select one sample for hold-out
                holdout_idx = np.random.randint(0, len(samples))
                
                # Add to hold-out set
                holdout_indices.append(samples[holdout_idx][0])
                # Ensure consistent IsNonTarget handling
                sample_conditions = samples[holdout_idx][1]['conditions']
                if 'IsNonTarget' not in sample_conditions:
                    sample_conditions['IsNonTarget'] = False
                
                holdout_info.append({
                    'category': category,
                    'concentration': concentration,
                    'filename': sample_conditions.get('Original_filename', 'Unknown'),
                    'is_nontarget': sample_conditions['IsNonTarget']
                })
                
                # Add rest to training set
                for i, (idx, _) in enumerate(samples):
                    if i != holdout_idx:
                        train_indices.append(idx)
                
                print(f"  Concentration {concentration:.3f}: {len(samples)} samples (1 to hold-out)")
    
    # Create final datasets
    train_data = [data_list[i] for i in train_indices]
    holdout_data = [data_list[i] for i in holdout_indices]
    
    print(f"\nSplit complete:")
    print(f"  Training set: {len(train_data)} samples")
    print(f"  Hold-out set: {len(holdout_data)} samples")
    
    # Count non-targets in hold-out
    n_nontargets = sum(1 for info in holdout_info if info['is_nontarget'])
    print(f"  Non-target samples in hold-out: {n_nontargets}")
    
    return train_data, holdout_data, holdout_info

class RegressionDataset(Dataset):
    """Dataset for direct regression from signal + conditions to concentration."""
    
    def __init__(self, data_list, scalers=None, concentration_scaler=None, fit_scalers=True, scaler_type='standard'):
        """
        Args:
            data_list: List of data dictionaries
            scalers: Dictionary of scalers for signal normalization
            concentration_scaler: Scaler for concentration normalization
            fit_scalers: Whether to fit scalers on this data
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.data_list = data_list
        self.scaler_type = scaler_type
        
        # Process signal data
        self.signal_data = []
        self.condition_features = []
        self.concentrations = []
        self.filenames = []
        self.targets = []  # Store target for plotting
        self.setups = [] # Store setup for plotting
        self.is_nontarget = []  # Store non-target flag for plotting
        
        # Extract all signal data first
        all_cycles = []
        for item in data_list:
            cycles = np.array(item['cycles'])
            
            # Ensure consistent shape
            if cycles.shape != (200, 61, 3):
                if len(cycles) < 200:
                    padding = np.zeros((200 - len(cycles), 61, 3))
                    cycles = np.vstack([cycles, padding])
                elif len(cycles) > 200:
                    cycles = cycles[:200]
                
                if cycles.shape[1] != 61:
                    print(f"Warning: Unexpected number of points per cycle: {cycles.shape[1]}")
            
            all_cycles.append(cycles)
        
        # Convert to numpy array
        all_cycles = np.array(all_cycles)
        print(f"Raw signal data shape: {all_cycles.shape}")
        
        # Setup scalers for signal normalization
        if scalers is None:
            if scaler_type == 'standard':
                self.scalers = {
                    'GateI': StandardScaler(),
                    'GateV': StandardScaler(),
                    'DrainI': StandardScaler()
                }
            elif scaler_type == 'minmax':
                self.scalers = {
                    'GateI': MinMaxScaler(),
                    'GateV': MinMaxScaler(),
                    'DrainI': MinMaxScaler()
                }
            elif scaler_type == 'robust':
                from sklearn.preprocessing import RobustScaler
                self.scalers = {
                    'GateI': RobustScaler(),
                    'GateV': RobustScaler(),
                    'DrainI': RobustScaler()
                }
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
        else:
            self.scalers = scalers
        
        # Fit scalers if needed
        if fit_scalers:
            print(f"Fitting {scaler_type} scalers...")
            
            # Extract each dimension for fitting
            gatei_all = all_cycles[:, :, :, 0].flatten()
            gatev_all = all_cycles[:, :, :, 1].flatten()
            draini_all = all_cycles[:, :, :, 2].flatten()
            
            print(f"GateI stats: min={gatei_all.min():.2e}, max={gatei_all.max():.2e}, mean={gatei_all.mean():.2e}")
            print(f"GateV stats: min={gatev_all.min():.2e}, max={gatev_all.max():.2e}, mean={gatev_all.mean():.2e}")
            print(f"DrainI stats: min={draini_all.min():.2e}, max={draini_all.max():.2e}, mean={draini_all.mean():.2e}")
            
            # Fit scalers
            self.scalers['GateI'].fit(gatei_all.reshape(-1, 1))
            self.scalers['GateV'].fit(gatev_all.reshape(-1, 1))
            self.scalers['DrainI'].fit(draini_all.reshape(-1, 1))
        
        # Normalize signal data
        normalized_cycles = np.zeros_like(all_cycles)
        for i, cycles in enumerate(all_cycles):
            # Normalize each dimension
            gatei_norm = self.scalers['GateI'].transform(cycles[:, :, 0].flatten().reshape(-1, 1))
            gatev_norm = self.scalers['GateV'].transform(cycles[:, :, 1].flatten().reshape(-1, 1))
            draini_norm = self.scalers['DrainI'].transform(cycles[:, :, 2].flatten().reshape(-1, 1))
            
            # Reshape back
            normalized_cycles[i, :, :, 0] = gatei_norm.reshape(cycles.shape[0], cycles.shape[1])
            normalized_cycles[i, :, :, 1] = gatev_norm.reshape(cycles.shape[0], cycles.shape[1])
            normalized_cycles[i, :, :, 2] = draini_norm.reshape(cycles.shape[0], cycles.shape[1])
        
        # Flatten signal data for input to network
        self.signal_data = normalized_cycles.reshape(len(all_cycles), -1)
        print(f"Normalized signal data shape: {self.signal_data.shape}")
        
        # Process experimental conditions and targets
        self.target_mapping = {'Bacteria': 0, 'HRP': 1, 'IgG': 2}
        self.setup_mapping = {'HRPAB': 0, 'Control': 1, 'Au40': 2}
        
        for item in data_list:
            conditions = item['conditions']
            
            # Ensure IsNonTarget field is consistently set
            if 'IsNonTarget' not in conditions:
                conditions['IsNonTarget'] = False
            
            # Extract target one-hot (3 dimensions: Bacteria, HRP, IgG)
            target = conditions.get('Target', 'Unknown')
            target_onehot = [0, 0, 0]
            if target in self.target_mapping:
                target_onehot[self.target_mapping[target]] = 1
            
            # Extract setup one-hot (3 dimensions: HRPAB, Control, Au40)
            setup = conditions.get('Setup', 'Unknown')
            setup_onehot = [0, 0, 0]
            if setup in self.setup_mapping:
                setup_onehot[self.setup_mapping[setup]] = 1
            
            # Extract non-target flag (1 dimension)
            is_nontarget = 1 if conditions['IsNonTarget'] else 0
            
            # Combine all condition features
            condition_features = target_onehot + setup_onehot + [is_nontarget]
            self.condition_features.append(condition_features)
            
            # Extract concentration (target variable)
            concentration = conditions.get('Concentration_transformed', 0.0)
            
            # DEBUG: Track NonTarget concentration values during dataset creation
            if conditions['IsNonTarget']:
                print(f"DEBUG DATASET: NonTarget sample {conditions.get('Original_filename', 'Unknown')}: "
                      f"Target={target}, Concentration_transformed={concentration}")
                if concentration != -19.0:
                    print(f"  ❌ ERROR: NonTarget sample has concentration != -19: {concentration}")
            
            self.concentrations.append(concentration)
            
            # Store additional info for plotting
            self.filenames.append(conditions.get('Original_filename', 'Unknown'))
            self.targets.append(target)
            self.setups.append(conditions.get('Setup', 'Unknown'))
            self.is_nontarget.append(is_nontarget)
        
        # Normalize concentration targets
        concentrations_array = np.array(self.concentrations)
        if fit_scalers:
            self.concentration_scaler = StandardScaler()
            concentrations_normalized = self.concentration_scaler.fit_transform(concentrations_array.reshape(-1, 1)).flatten()
            print(f"TRAINING SET - Concentration stats BEFORE normalization: min={concentrations_array.min():.2f}, max={concentrations_array.max():.2f}, mean={concentrations_array.mean():.2f}, std={concentrations_array.std():.2f}")
            print(f"TRAINING SET - Concentration stats AFTER normalization: min={concentrations_normalized.min():.2f}, max={concentrations_normalized.max():.2f}, mean={concentrations_normalized.mean():.2f}, std={concentrations_normalized.std():.2f}")
        else:
            self.concentration_scaler = concentration_scaler
            concentrations_normalized = self.concentration_scaler.transform(concentrations_array.reshape(-1, 1)).flatten()
            print(f"HOLDOUT SET - Concentration stats BEFORE normalization: min={concentrations_array.min():.2f}, max={concentrations_array.max():.2f}, mean={concentrations_array.mean():.2f}")
            print(f"HOLDOUT SET - Concentration stats AFTER normalization: min={concentrations_normalized.min():.2f}, max={concentrations_normalized.max():.2f}, mean={concentrations_normalized.mean():.2f}")
        
        # Convert to tensors
        self.signal_data = torch.tensor(self.signal_data, dtype=torch.float32)
        self.condition_features = torch.tensor(self.condition_features, dtype=torch.float32)
        self.concentrations = torch.tensor(concentrations_normalized, dtype=torch.float32)
        
        print(f"Dataset created with {len(self.signal_data)} samples")
        print(f"Signal feature dimension: {self.signal_data.shape[1]}")
        print(f"Condition feature dimension: {self.condition_features.shape[1]}")
        print(f"Normalized concentration range: {self.concentrations.min():.2f} to {self.concentrations.max():.2f}")
        
        # Print condition feature statistics
        print("\nCondition feature statistics:")
        print(f"Target distribution: {self.condition_features[:, :3].sum(dim=0)}")
        print(f"Setup distribution: {self.condition_features[:, 3:6].sum(dim=0)}")
        print(f"Non-target samples: {self.condition_features[:, 6].sum().item()}")
    
    def __len__(self):
        return len(self.signal_data)
    
    def __getitem__(self, idx):
        return (
            self.signal_data[idx], 
            self.condition_features[idx], 
            self.concentrations[idx],
            self.filenames[idx],
            self.targets[idx],
            self.setups[idx],
            self.is_nontarget[idx]
        )

class ConcentrationRegressor(nn.Module):
    """Neural network for direct regression to concentration."""
    
    def __init__(self, signal_dim, condition_dim, hidden_dims=[1024, 512, 256], dropout_rate=0.02):
        super(ConcentrationRegressor, self).__init__()
        
        # Signal processing branch
        signal_layers = []
        current_dim = signal_dim
        
        for hidden_dim in hidden_dims:
            signal_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        self.signal_processor = nn.Sequential(*signal_layers)
        
        # Condition processing branch
        condition_layers = [
            nn.Linear(condition_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate / 2)
        ]
        self.condition_processor = nn.Sequential(*condition_layers)
        
        # Combined regression head
        combined_dim = current_dim + 32
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
    def forward(self, signal, conditions):
        # Process signal
        signal_features = self.signal_processor(signal)
        
        # Process conditions
        condition_features = self.condition_processor(conditions)
        
        # Combine features
        combined = torch.cat([signal_features, condition_features], dim=1)
        
        # Predict concentration
        concentration = self.regressor(combined)
        
        return concentration.squeeze()

class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=100, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def restore_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

def create_sklearn_model(model_name, random_state=42):
    """Create sklearn-based models."""
    if model_name == 'rf':
        return RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    elif model_name == 'gbdt':
        return GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    elif model_name == 'svr':
        return SVR(kernel='rbf', C=1.0, epsilon=0.1)
    elif model_name == 'knr':
        return KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    elif model_name == 'xgboost':
        if XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        else:
            raise ImportError("XGBoost not available")
    elif model_name == 'catboost':
        if CATBOOST_AVAILABLE:
            return cb.CatBoostRegressor(iterations=100, random_state=random_state, verbose=False)
        else:
            raise ImportError("CatBoost not available")
    elif model_name == 'lightgbm':
        if LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1)
        else:
            raise ImportError("LightGBM not available")
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_sklearn_model(model, X_train, y_train, X_val, y_val):
    """Train sklearn models."""
    print(f"Training {type(model).__name__}...")
    model.fit(X_train, y_train)
    
    # Simple validation
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_loss = mean_squared_error(y_train, train_pred)
    val_loss = mean_squared_error(y_val, val_pred)
    
    print(f"Training MSE: {train_loss:.6f}, Validation MSE: {val_loss:.6f}")
    
    return [train_loss], [val_loss]  # Return as lists for consistency

def evaluate_sklearn_model(model, X_test, y_test, concentration_scaler, denormalize=True):
    """Evaluate sklearn models."""
    predictions = model.predict(X_test)
    
    if denormalize:
        predictions = concentration_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        y_test = concentration_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    return predictions, y_test

def prepare_sklearn_data(dataset):
    """Prepare data for sklearn models."""
    # Combine signal and condition features
    X = torch.cat([dataset.signal_data, dataset.condition_features], dim=1).numpy()
    y = dataset.concentrations.numpy()
    
    # Also return metadata
    filenames = dataset.filenames
    targets = dataset.targets
    setups = dataset.setups
    is_nontarget = dataset.is_nontarget
    
    return X, y, filenames, targets, setups, is_nontarget

def focal_loss_for_regression(predictions, targets, alpha=0.25, gamma=2.0):
    """
    Focal loss for regression - focuses on hard examples.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        alpha: Weighting factor (0-1)
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    # Calculate absolute errors
    abs_errors = torch.abs(predictions - targets)
    
    # Normalize errors to [0, 1] range for focal weighting
    max_error = abs_errors.max().detach() + 1e-8
    normalized_errors = abs_errors / max_error
    
    # Calculate focal weight: (1 - pt)^gamma where pt is "correctness"
    # For regression, we use (1 - normalized_error) as "correctness"
    pt = 1 - normalized_errors
    focal_weight = alpha * (1 - pt) ** gamma
    
    # Apply focal weighting to MSE loss
    mse_loss = (predictions - targets) ** 2
    focal_mse = focal_weight * mse_loss
    
    return focal_mse.mean()

def train_model(model, train_loader, val_loader, num_epochs=500, learning_rate=1e-3, device='mps', 
                use_focal_loss=False, use_gradient_clipping=False, grad_clip_value=1.0):
    """Train the regression model."""
    
    # Select loss function
    if use_focal_loss:
        def criterion(pred, target):
            return focal_loss_for_regression(pred, target, alpha=0.25, gamma=2.0)
        print("Using focal loss for regression")
    else:
        def criterion(pred, target):
            return nn.functional.mse_loss(pred, target)
        print("Using standard MSE loss")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    early_stopping = EarlyStopping(patience=100)
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (signal, conditions, concentration, _, _, _, _) in enumerate(train_loader):
            signal, conditions, concentration = signal.to(device), conditions.to(device), concentration.to(device)
            
            optimizer.zero_grad()
            predictions = model(signal, conditions)
            loss = criterion(predictions, concentration)
            loss.backward()
            
            # Apply gradient clipping if enabled
            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase (always use standard MSE for validation)
        model.eval()
        val_loss = 0.0
        val_criterion = nn.MSELoss()
        
        with torch.no_grad():
            for signal, conditions, concentration, _, _, _, _ in val_loader:
                signal, conditions, concentration = signal.to(device), conditions.to(device), concentration.to(device)
                predictions = model(signal, conditions)
                loss = val_criterion(predictions, concentration)
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Restore best model
    early_stopping.restore_best_model(model)
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, concentration_scaler, device='cpu', denormalize=True):
    """Evaluate the model and return predictions and ground truth."""
    
    model.eval()
    predictions = []
    ground_truth = []
    filenames_list = []
    targets_list = []
    setups_list = []
    is_nontarget_list = []
    
    with torch.no_grad():
        for signal, conditions, concentration, filenames, targets, setups, is_nontarget in test_loader:
            signal, conditions, concentration = signal.to(device), conditions.to(device), concentration.to(device)
            pred = model(signal, conditions)
            
            predictions.extend(pred.cpu().numpy())
            ground_truth.extend(concentration.cpu().numpy())
            filenames_list.extend(filenames)
            targets_list.extend(targets)
            setups_list.extend(setups)
            is_nontarget_list.extend(is_nontarget) # This is already a list of ints
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Denormalize to original scale for meaningful metrics
    if denormalize:
        print(f"DEBUG DENORMALIZE: Before denormalization - predictions range: {predictions.min():.3f} to {predictions.max():.3f}")
        print(f"DEBUG DENORMALIZE: Before denormalization - ground_truth range: {ground_truth.min():.3f} to {ground_truth.max():.3f}")
        
        predictions = concentration_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        ground_truth = concentration_scaler.inverse_transform(ground_truth.reshape(-1, 1)).flatten()
        
        print(f"DEBUG DENORMALIZE: After denormalization - predictions range: {predictions.min():.3f} to {predictions.max():.3f}")
        print(f"DEBUG DENORMALIZE: After denormalization - ground_truth range: {ground_truth.min():.3f} to {ground_truth.max():.3f}")
        
        # Check for any ground truth values that should be -19 but aren't
        expected_minus19_mask = (ground_truth > -19.1) & (ground_truth < -18.9)  # Close to -19
        not_minus19_mask = ~expected_minus19_mask & (ground_truth != -19.0)
        if not_minus19_mask.any():
            print(f"DEBUG DENORMALIZE: Found {not_minus19_mask.sum()} ground truth values that are not -19:")
            problematic_values = ground_truth[not_minus19_mask]
            for i, val in enumerate(problematic_values[:5]):  # Show first 5
                print(f"  Ground truth #{i}: {val}")
    
    return predictions, ground_truth, filenames_list, targets_list, setups_list, is_nontarget_list

def calculate_metrics(predictions, ground_truth):
    """Calculate regression metrics."""
    
    r2 = r2_score(ground_truth, predictions)
    mse = mean_squared_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ground_truth, predictions)
    
    # Calculate Pearson correlation
    corr, p_value = stats.pearsonr(ground_truth, predictions)
    
    metrics = {
        'r2_score': float(r2),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson_corr': float(corr),
        'p_value': float(p_value)
    }
    
    return metrics

def plot_modified_scatter_plot(df, output_dir, shuffle_idx=None):
    """Create scatter plots treating NonTarget as regular points, with split IgG categories."""
    
    # Separate train and test data
    train_df = df[df['dataset'] == 'train']
    test_df = df[df['dataset'] == 'test']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.4, wspace=0.25)
    
    ax_full = fig.add_subplot(gs[0, :])
    ax_igg_au40 = fig.add_subplot(gs[1, 0])
    ax_igg_hrpab = fig.add_subplot(gs[1, 1])
    ax_hrp = fig.add_subplot(gs[2, 0])
    ax_bacteria = fig.add_subplot(gs[2, 1])
    
    categories = {
        'IgG-Au40': {'ax': ax_igg_au40, 'marker': 's', 'name': 'IgG-Au40'},
        'IgG-HRPAB': {'ax': ax_igg_hrpab, 'marker': 'P', 'name': 'IgG-HRPAB'},
        'HRP': {'ax': ax_hrp, 'marker': 'o', 'name': 'HRP'},
        'Bacteria': {'ax': ax_bacteria, 'marker': '^', 'name': 'Bacteria'}
    }
    train_color, test_color = 'blue', 'red'

    # --- Full Plot ---
    for name, props in categories.items():
        train_cat = train_df[train_df['category'] == name]
        test_cat = test_df[test_df['category'] == name]
        
        if not train_cat.empty:
            ax_full.scatter(train_cat['ground_truth'], train_cat['predictions'], alpha=0.7, s=80, color=train_color, marker=props['marker'], label=f'{name} Training', edgecolors='black', linewidth=0.5)
        if not test_cat.empty:
            ax_full.scatter(test_cat['ground_truth'], test_cat['predictions'], alpha=0.9, s=100, color=test_color, marker=props['marker'], label=f'{name} Test', edgecolors='white', linewidth=0.8)

    min_val, max_val = min(df['ground_truth'].min(), df['predictions'].min()), max(df['ground_truth'].max(), df['predictions'].max())
    ax_full.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    ax_full.set_xlabel('Ground Truth: log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=16)
    ax_full.set_ylabel('Predicted: log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=16)
    title = f'Full Plot: Predicted vs Ground Truth (Split {shuffle_idx + 1})' if shuffle_idx is not None else 'Full Plot'
    ax_full.set_title(title, fontsize=22, pad=20)
    ax_full.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=14)
    ax_full.grid(True, alpha=0.3)

    # --- Category Subplots ---
    for name, props in categories.items():
        ax, marker = props['ax'], props['marker']
        train_cat = train_df[train_df['category'] == name]
        test_cat = test_df[test_df['category'] == name]

        if not train_cat.empty:
            ax.scatter(train_cat['ground_truth'], train_cat['predictions'], alpha=0.7, s=80, color=train_color, marker=marker, label='Training', edgecolors='black', linewidth=0.5)
        if not test_cat.empty:
            ax.scatter(test_cat['ground_truth'], test_cat['predictions'], alpha=0.9, s=100, color=test_color, marker=marker, label='Test', edgecolors='white', linewidth=0.8)
        
        cat_df = df[df['category'] == name]
        if not cat_df.empty:
            cat_min, cat_max = min(cat_df['ground_truth'].min(), cat_df['predictions'].min()), max(cat_df['ground_truth'].max(), cat_df['predictions'].max())
            ax.plot([cat_min, cat_max], [cat_min, cat_max], 'k--', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Ground Truth: log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=14)
        ax.set_ylabel('Predicted: log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=14)
        ax.set_title(f'{name}', fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plot_name = f'modified_scatter_plot_split_{shuffle_idx + 1}.png' if shuffle_idx is not None else 'modified_scatter_plot.png'
    plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Modified scatter plot saved to: {os.path.join(output_dir, plot_name)}")

def plot_error_bar_version_a(df, output_dir, shuffle_idx=None):
    """Error bar plot (Version A): Mean ± Std from all data (train + test)."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=False, sharey=False)
    axes = axes.flatten()
    categories = {
        'IgG-Au40': {'marker': 's', 'color': 'purple'}, 'IgG-HRPAB': {'marker': 'P', 'color': 'blue'},
        'HRP': {'marker': 'o', 'color': 'green'}, 'Bacteria': {'marker': '^', 'color': 'red'}
    }

    for ax, (name, props) in zip(axes, categories.items()):
        cat_df = df[df['category'] == name]
        if not cat_df.empty:
            stats = cat_df.groupby('ground_truth')['predictions'].agg(['mean', 'std']).reset_index()
            stats['std'] = stats['std'].fillna(0)
            ax.errorbar(stats['ground_truth'], stats['mean'], yerr=stats['std'], fmt=props['marker'], color=props['color'], markersize=12, capsize=5, label=f'{name} (Mean ± Std)')
            
            # Add y=x line
            all_vals = pd.concat([stats['ground_truth'], stats['mean']])
            min_val, max_val = all_vals.min(), all_vals.max()
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Prediction')

        ax.set_xlabel('Ground Truth: log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=14)
        ax.set_ylabel('Predicted: log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=14)
        ax.set_title(f'{name}', fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Error Bar Plot (Version A: All Data) - Split {shuffle_idx + 1}' if shuffle_idx is not None else 'Error Bar Plot (Version A)', fontsize=22, y=1.02)
    plt.tight_layout()
    plot_name = f'error_bar_plot_vA_split_{shuffle_idx + 1}.png' if shuffle_idx is not None else 'error_bar_plot_vA.png'
    plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error bar plot (vA) saved to: {os.path.join(output_dir, plot_name)}")

def plot_error_bar_version_b(df, output_dir, shuffle_idx=None):
    """Error bar plot (Version B): Mean ± Std from training data, test points overlaid."""
    train_df = df[df['dataset'] == 'train']
    test_df = df[df['dataset'] == 'test']
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=False, sharey=False)
    axes = axes.flatten()
    categories = {
        'IgG-Au40': {'marker': 's', 'color': 'purple'}, 'IgG-HRPAB': {'marker': 'P', 'color': 'blue'},
        'HRP': {'marker': 'o', 'color': 'green'}, 'Bacteria': {'marker': '^', 'color': 'red'}
    }

    for ax, (name, props) in zip(axes, categories.items()):
        train_cat = train_df[train_df['category'] == name]
        test_cat = test_df[test_df['category'] == name]

        if not train_cat.empty:
            stats = train_cat.groupby('ground_truth')['predictions'].agg(['mean', 'std']).reset_index()
            stats['std'] = stats['std'].fillna(0)
            ax.errorbar(stats['ground_truth'], stats['mean'], yerr=stats['std'], fmt=props['marker'], color=props['color'], markersize=12, capsize=5, label='Training (Mean ± Std)', alpha=0.7)
        if not test_cat.empty:
            ax.scatter(test_cat['ground_truth'], test_cat['predictions'], color='red', marker='x', s=150, linewidths=2, label='Test Points', alpha=0.9)
        
        # Add y=x line
        all_vals_truth = pd.concat([train_cat['ground_truth'], test_cat['ground_truth']])
        all_vals_pred = pd.concat([train_cat['predictions'], test_cat['predictions']])
        if not all_vals_truth.empty:
            min_val = min(all_vals_truth.min(), all_vals_pred.min())
            max_val = max(all_vals_truth.max(), all_vals_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Prediction')

        ax.set_xlabel('Ground Truth: log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=14)
        ax.set_ylabel('Predicted: log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=14)
        ax.set_title(f'{name}', fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Error Bar Plot (Version B: Train/Test Split) - Split {shuffle_idx + 1}' if shuffle_idx is not None else 'Error Bar Plot (Version B)', fontsize=22, y=1.02)
    plt.tight_layout()
    plot_name = f'error_bar_plot_vB_split_{shuffle_idx + 1}.png' if shuffle_idx is not None else 'error_bar_plot_vB.png'
    plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error bar plot (vB) saved to: {os.path.join(output_dir, plot_name)}")

def plot_error_bar_version_c(df, output_dir, shuffle_idx=None):
    """Error bar plot (Version C): Based on Version A with ML-LDL and Traditional LDL lines."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=False, sharey=False)
    axes = axes.flatten()
    categories = {
        'IgG-Au40': {'marker': 's', 'color': 'purple'}, 'IgG-HRPAB': {'marker': 'P', 'color': 'blue'},
        'HRP': {'marker': 'o', 'color': 'green'}, 'Bacteria': {'marker': '^', 'color': 'red'}
    }
    
    # Traditional LDL values (log-transformed)
    traditional_ldl = {
        'IgG-Au40': np.log10(1.23e-15 + 1e-19),  # 1.23 fM
        'IgG-HRPAB': np.log10(2.3e-14 + 1e-19),  # 23.4 fM
        'HRP': np.log10(8.5e-17 + 1e-19),        # 84.8 aM
        'Bacteria': np.log10(1.2e-15 + 1e-19)    # 1.2 × 10⁻¹⁵ M
    }

    for ax, (name, props) in zip(axes, categories.items()):
        cat_df = df[df['category'] == name]
        if not cat_df.empty:
            stats = cat_df.groupby('ground_truth')['predictions'].agg(['mean', 'std']).reset_index()
            stats['std'] = stats['std'].fillna(0)
            ax.errorbar(stats['ground_truth'], stats['mean'], yerr=stats['std'], fmt=props['marker'], color=props['color'], markersize=12, capsize=5, label=f'{name} (Mean ± Std)')
            
            # Add y=x line
            all_vals = pd.concat([stats['ground_truth'], stats['mean']])
            min_val, max_val = all_vals.min(), all_vals.max()
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Prediction')
            
            # Calculate ML-LDL (mean + 3*std at lowest concentration)
            lowest_conc = stats['ground_truth'].min()
            lowest_stats = stats[stats['ground_truth'] == lowest_conc]
            if not lowest_stats.empty:
                ml_ldl = lowest_stats['mean'].iloc[0] + 3 * lowest_stats['std'].iloc[0]
                ax.axhline(y=ml_ldl, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'ML-LDL: {ml_ldl:.2f}')
            
            # Add Traditional LDL line
            if name in traditional_ldl:
                trad_ldl = traditional_ldl[name]
                ax.axhline(y=trad_ldl, color='gray', linestyle='--', linewidth=2, alpha=0.8, label=f'Traditional LDL: {trad_ldl:.2f}')

        ax.set_xlabel('Ground Truth: log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=14)
        ax.set_ylabel('Predicted: log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=14)
        ax.set_title(f'{name}', fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Error Bar Plot (Version C: All Data + LDL Lines) - Split {shuffle_idx + 1}' if shuffle_idx is not None else 'Error Bar Plot (Version C: All Data + LDL Lines)', fontsize=22, y=1.02)
    plt.tight_layout()
    plot_name = f'error_bar_plot_vC_split_{shuffle_idx + 1}.png' if shuffle_idx is not None else 'error_bar_plot_vC.png'
    plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error bar plot (vC) saved to: {os.path.join(output_dir, plot_name)}")

def plot_error_bar_version_d(df, output_dir, shuffle_idx=None):
    """Error bar plot (Version D): Based on Version B with ML-LDL and Traditional LDL lines."""
    train_df = df[df['dataset'] == 'train']
    test_df = df[df['dataset'] == 'test']
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=False, sharey=False)
    axes = axes.flatten()
    categories = {
        'IgG-Au40': {'marker': 's', 'color': 'purple'}, 'IgG-HRPAB': {'marker': 'P', 'color': 'blue'},
        'HRP': {'marker': 'o', 'color': 'green'}, 'Bacteria': {'marker': '^', 'color': 'red'}
    }
    
    # Traditional LDL values (log-transformed)
    traditional_ldl = {
        'IgG-Au40': np.log10(1.23e-15 + 1e-19),  # 1.23 fM
        'IgG-HRPAB': np.log10(2.3e-14 + 1e-19),  # 23.4 fM
        'HRP': np.log10(8.5e-17 + 1e-19),        # 84.8 aM
        'Bacteria': np.log10(1.2e-15 + 1e-19)    # 1.2 × 10⁻¹⁵ M
    }

    for ax, (name, props) in zip(axes, categories.items()):
        train_cat = train_df[train_df['category'] == name]
        test_cat = test_df[test_df['category'] == name]

        if not train_cat.empty:
            stats = train_cat.groupby('ground_truth')['predictions'].agg(['mean', 'std']).reset_index()
            stats['std'] = stats['std'].fillna(0)
            ax.errorbar(stats['ground_truth'], stats['mean'], yerr=stats['std'], fmt=props['marker'], color=props['color'], markersize=12, capsize=5, label='Training (Mean ± Std)', alpha=0.7)
            
            # Calculate ML-LDL (mean + 3*std at lowest concentration)
            lowest_conc = stats['ground_truth'].min()
            lowest_stats = stats[stats['ground_truth'] == lowest_conc]
            if not lowest_stats.empty:
                ml_ldl = lowest_stats['mean'].iloc[0] + 3 * lowest_stats['std'].iloc[0]
                ax.axhline(y=ml_ldl, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'ML-LDL: {ml_ldl:.2f}')
        
        if not test_cat.empty:
            ax.scatter(test_cat['ground_truth'], test_cat['predictions'], color='red', marker='x', s=150, linewidths=2, label='Test Points', alpha=0.9)
        
        # Add y=x line
        all_vals_truth = pd.concat([train_cat['ground_truth'], test_cat['ground_truth']])
        all_vals_pred = pd.concat([train_cat['predictions'], test_cat['predictions']])
        if not all_vals_truth.empty:
            min_val = min(all_vals_truth.min(), all_vals_pred.min())
            max_val = max(all_vals_truth.max(), all_vals_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Prediction')
        
        # Add Traditional LDL line
        if name in traditional_ldl:
            trad_ldl = traditional_ldl[name]
            ax.axhline(y=trad_ldl, color='gray', linestyle='--', linewidth=2, alpha=0.8, label=f'Traditional LDL: {trad_ldl:.2f}')

        ax.set_xlabel('Ground Truth: log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=14)
        ax.set_ylabel('Predicted: log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=14)
        ax.set_title(f'{name}', fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Error Bar Plot (Version D: Train/Test Split + LDL Lines) - Split {shuffle_idx + 1}' if shuffle_idx is not None else 'Error Bar Plot (Version D: Train/Test Split + LDL Lines)', fontsize=22, y=1.02)
    plt.tight_layout()
    plot_name = f'error_bar_plot_vD_split_{shuffle_idx + 1}.png' if shuffle_idx is not None else 'error_bar_plot_vD.png'
    plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error bar plot (vD) saved to: {os.path.join(output_dir, plot_name)}")

def plot_sigma_error_analysis(df, output_dir, shuffle_idx=None):
    """Create sigma error level analysis plot for a specific split."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=False)
    axes = axes.flatten()
    categories = {
        'IgG-Au40': {'color': 'purple'}, 'IgG-HRPAB': {'color': 'blue'},
        'HRP': {'color': 'green'}, 'Bacteria': {'color': 'red'}
    }
    split_sigma_stats = {}

    for ax, (name, props) in zip(axes, categories.items()):
        cat_df = df[df['category'] == name]
        if not cat_df.empty:
            stats = cat_df.groupby('ground_truth')['predictions'].agg(['std', 'count']).reset_index()
            stats = stats[stats['count'] > 1].rename(columns={'std': 'sigma'})
            
            if not stats.empty:
                bars = ax.bar(stats['ground_truth'], stats['sigma'], color=props['color'], alpha=0.7, edgecolor='black')
                for bar, n, sigma_val in zip(bars, stats['count'], stats['sigma']):
                    height = bar.get_height()
                    # Add sigma value on top of bar
                    ax.text(bar.get_x() + bar.get_width() / 2., height, f'{sigma_val:.3f}', ha='center', va='bottom', fontsize=11, color='black', fontweight='bold')
                
                avg_sigma = np.average(stats['sigma'], weights=stats['count'])
                ax.axhline(y=avg_sigma, color='red', linestyle='--', label=f'Weighted Avg σ: {avg_sigma:.3f}')
                split_sigma_stats[name] = {'weighted_avg_sigma': avg_sigma, 'stats': stats.to_dict('records')}
        
        ax.set_xlabel('log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=14)
        ax.set_ylabel('Prediction Standard Deviation (σ)', fontsize=14)
        ax.set_title(f'{name}', fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Sigma Error Analysis - Split {shuffle_idx + 1}' if shuffle_idx is not None else 'Sigma Error Analysis', fontsize=22, y=1.02)
    plt.tight_layout()
    plot_name = f'sigma_error_analysis_split_{shuffle_idx + 1}.png' if shuffle_idx is not None else 'sigma_error_analysis.png'
    plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sigma analysis plot saved to: {os.path.join(output_dir, plot_name)}")
    return split_sigma_stats

def plot_3sigma_error_analysis(df, output_dir, shuffle_idx=None):
    """Create 3-sigma error level analysis plot for a specific split."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=False)
    axes = axes.flatten()
    categories = {
        'IgG-Au40': {'color': 'purple'}, 'IgG-HRPAB': {'color': 'blue'},
        'HRP': {'color': 'green'}, 'Bacteria': {'color': 'red'}
    }

    for ax, (name, props) in zip(axes, categories.items()):
        cat_df = df[df['category'] == name]
        if not cat_df.empty:
            stats = cat_df.groupby('ground_truth')['predictions'].agg(['std', 'count']).reset_index()
            stats = stats[stats['count'] > 1].rename(columns={'std': 'sigma'})
            stats['3sigma'] = stats['sigma'] * 3  # Calculate 3-sigma
            
            if not stats.empty:
                bars = ax.bar(stats['ground_truth'], stats['3sigma'], color=props['color'], alpha=0.7, edgecolor='black')
                for bar, three_sigma_val in zip(bars, stats['3sigma']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height, f'{three_sigma_val:.3f}', ha='center', va='bottom', fontsize=11, color='black', fontweight='bold')
        
        ax.set_xlabel('log₁₀(Concentration + 10⁻¹⁹) [mol/L]', fontsize=14)
        ax.set_ylabel('Prediction 3-Sigma (3*σ)', fontsize=14)
        ax.set_title(f'{name}', fontsize=18)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'3-Sigma Error Analysis - Split {shuffle_idx + 1}' if shuffle_idx is not None else '3-Sigma Error Analysis', fontsize=22, y=1.02)
    plt.tight_layout()
    plot_name = f'3sigma_error_analysis_split_{shuffle_idx + 1}.png' if shuffle_idx is not None else '3sigma_error_analysis.png'
    plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3-Sigma analysis plot saved to: {os.path.join(output_dir, plot_name)}")

def generate_cross_split_summary(all_sigma_stats, output_dir):
    """Generate summary plot of weighted average sigma across all splits."""
    summary_dir = os.path.join(output_dir, "cross_split_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    categories = ['IgG-Au40', 'IgG-HRPAB', 'HRP', 'Bacteria']
    colors = ['purple', 'blue', 'green', 'red']
    
    summary_data = defaultdict(list)
    split_nums = sorted(all_sigma_stats.keys())
    for split_num in split_nums:
        for cat in categories:
            summary_data[cat].append(all_sigma_stats[split_num].get(cat, {}).get('weighted_avg_sigma', np.nan))

    for cat, color in zip(categories, colors):
        ax.plot(split_nums, summary_data[cat], 'o-', color=color, label=cat, markersize=10, linewidth=2.5)

    ax.set_xlabel('Split Number', fontsize=14)
    ax.set_ylabel('Weighted Average σ', fontsize=14)
    ax.set_title('Weighted Average Sigma Across All Splits', fontsize=18)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(split_nums)
    
    plt.tight_layout()
    plot_path = os.path.join(summary_dir, 'cross_split_sigma_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cross-split sigma summary saved to: {plot_path}")

def plot_training_history(train_losses, val_losses, output_dir, split_idx=None):
    """Plot training and validation loss curves."""
    
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    title = 'Training and Validation Loss'
    if split_idx is not None:
        title += f' (Split {split_idx + 1})'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if split_idx is not None:
        plot_name = f'training_history_split_{split_idx + 1}.png'
    else:
        plot_name = 'training_history.png'
    
    plot_path = os.path.join(output_dir, plot_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to: {plot_path}")

def save_results(train_predictions, train_ground_truth, train_filenames, train_targets, train_setups, train_nontargets,
                test_predictions, test_ground_truth, test_filenames, test_targets, test_setups, test_nontargets,
                output_dir, split_idx=None):
    """Save results to JSON and CSV files with category-specific metrics."""

    # Create a 'category' column for easier grouping
    def get_category(row):
        if row['target'] == 'IgG':
            return f"IgG-{row['setup']}"
        elif row['target'] == 'HRP':
            return 'HRP' # Assuming HRP is always Control setup
        else: # Bacteria
            return 'Bacteria'
    
    # Create training results DataFrame
    train_df = pd.DataFrame({
        'dataset': 'train',
        'filename': train_filenames,
        'target': train_targets,
        'setup': train_setups,
        'is_nontarget': train_nontargets,
        'ground_truth': train_ground_truth,
        'predictions': train_predictions
    })
    train_df['category'] = train_df.apply(get_category, axis=1)
    train_df['absolute_error'] = np.abs(train_df['predictions'] - train_df['ground_truth'])
    train_df['squared_error'] = (train_df['predictions'] - train_df['ground_truth'])**2
    
    # Create test results DataFrame
    test_df = pd.DataFrame({
        'dataset': 'test',
        'filename': test_filenames,
        'target': test_targets,
        'setup': test_setups,
        'is_nontarget': test_nontargets,
        'ground_truth': test_ground_truth,
        'predictions': test_predictions
    })
    test_df['category'] = test_df.apply(get_category, axis=1)
    test_df['absolute_error'] = np.abs(test_df['predictions'] - test_df['ground_truth'])
    test_df['squared_error'] = (test_df['predictions'] - test_df['ground_truth'])**2
    
    # Combine results
    results_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Save to CSV
    if split_idx is not None:
        csv_name = f'predictions_split_{split_idx + 1}.csv'
    else:
        csv_name = 'predictions.csv'
    
    csv_path = os.path.join(output_dir, csv_name)
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # --- Metrics Calculation ---
    # Overall metrics
    train_metrics = calculate_metrics(train_predictions, train_ground_truth)
    test_metrics = calculate_metrics(test_predictions, test_ground_truth)
    
    # Calculate LDL and other stats for scatter plot y-predictions
    ldl_metrics = {}
    categories_for_ldl = ['full', 'IgG-Au40', 'IgG-HRPAB', 'HRP', 'Bacteria']

    for category in categories_for_ldl:
        if category == 'full':
            cat_df = results_df
        else:
            cat_df = results_df[results_df['category'] == category]

        if cat_df.empty:
            ldl_metrics[category] = {'error': 'No data for this category'}
            continue

        # Lowest concentration (-19 for non-targets)
        lowest_conc_points = cat_df[cat_df['is_nontarget'] == 1]
        
        lowest_conc_mean = None
        lowest_conc_sigma = None
        
        if not lowest_conc_points.empty:
            lowest_conc_mean = lowest_conc_points['predictions'].mean()
            lowest_conc_sigma = lowest_conc_points['predictions'].std()

        # Highest concentration
        regular_points = cat_df[cat_df['is_nontarget'] == 0]
        
        highest_conc_val = None
        highest_conc_mean = None
        highest_conc_sigma = None
        
        if not regular_points.empty:
            highest_conc_val = regular_points['ground_truth'].max()
            highest_conc_points = regular_points[regular_points['ground_truth'] == highest_conc_val]
            if not highest_conc_points.empty:
                highest_conc_mean = highest_conc_points['predictions'].mean()
                highest_conc_sigma = highest_conc_points['predictions'].std()
                
        # LDL noise ratio
        ldl_noise_ratio = None
        if lowest_conc_mean is not None and highest_conc_mean is not None:
            denominator = highest_conc_mean - lowest_conc_mean
            if abs(denominator) > 1e-9:
                ldl_noise_ratio = abs(lowest_conc_sigma / denominator)
        
        ldl_metrics[category] = {
            'lowest_concentration_value': -19.0,
            'lowest_concentration_predicted_mean': float(lowest_conc_mean) if lowest_conc_mean is not None else None,
            'lowest_concentration_predicted_sigma': float(lowest_conc_sigma) if lowest_conc_sigma is not None else None,
            'highest_concentration_value': float(highest_conc_val) if highest_conc_val is not None else None,
            'highest_concentration_predicted_mean': float(highest_conc_mean) if highest_conc_mean is not None else None,
            'highest_concentration_predicted_sigma': float(highest_conc_sigma) if highest_conc_sigma is not None else None,
            'LDL_noise_ratio': float(ldl_noise_ratio) if ldl_noise_ratio is not None else None,
        }
    
    # Calculate category-specific metrics
    category_metrics = {}
    categories = ['IgG-Au40', 'IgG-HRPAB', 'HRP', 'Bacteria']
    
    for category in categories:
        train_cat_df = train_df[train_df['category'] == category]
        test_cat_df = test_df[test_df['category'] == category]

        category_metrics[category] = {
            'n_samples': int(len(train_cat_df) + len(test_cat_df)),
            'n_train': int(len(train_cat_df)),
            'n_test': int(len(test_cat_df)),
        }
        
        # Combined metrics
        combined_df = pd.concat([train_cat_df, test_cat_df])
        if len(combined_df) > 1:
            category_metrics[category]['combined'] = calculate_metrics(combined_df['predictions'], combined_df['ground_truth'])

        # Train-specific metrics
        if len(train_cat_df) > 1:
            category_metrics[category]['train'] = calculate_metrics(train_cat_df['predictions'], train_cat_df['ground_truth'])

        # Test-specific metrics
        if len(test_cat_df) > 1:
            category_metrics[category]['test'] = calculate_metrics(test_cat_df['predictions'], test_cat_df['ground_truth'])
            
        # For Bacteria, also calculate metrics for non-target samples
        if category == 'Bacteria':
            nontarget_df = combined_df[combined_df['is_nontarget'] == 1]
            if len(nontarget_df) > 1:
                category_metrics[category]['nontarget_metrics'] = {
                    'n_samples': int(len(nontarget_df)),
                    **calculate_metrics(nontarget_df['predictions'], nontarget_df['ground_truth'])
                    }
    
    # Combine all metrics
    all_metrics = {
        'overall_train_metrics': train_metrics,
        'overall_test_metrics': test_metrics,
        'category_specific_metrics': category_metrics,
        'ldl_metrics': ldl_metrics
    }
    
    # Save metrics to JSON
    if split_idx is not None:
        metrics_name = f'metrics_split_{split_idx + 1}.json'
    else:
        metrics_name = 'metrics.json'
    
    metrics_path = os.path.join(output_dir, metrics_name)
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    print(f"Metrics saved to: {metrics_path}")
    
    return results_df

def main(holdout_type, model_name, algorithms=None):
    """Main function to run the direct regression experiment."""
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Direct regression for molecular sensing data using various ML algorithms.")
    
    # Model selection
    parser.add_argument('--algorithms', type=str, default='mlp', 
                        help="Comma-separated list of algorithms to run: mlp,rf,gbdt,svr,knr,xgboost,catboost,lightgbm")
    
    # Hyperparameters for MLP model (set to best known config)
    parser.add_argument('--hidden_dims', type=str, default='512,256,128', 
                        help="Hidden dimensions for MLP, comma-separated.")
    parser.add_argument('--dropout_rate', type=float, default=0.18, 
                        help="Dropout rate for MLP.")
    parser.add_argument('--learning_rate', type=float, default=0.003, 
                        help="Learning rate for training.")
    parser.add_argument('--batch_size', type=int, default=16, 
                        help="Batch size for training.")
    parser.add_argument('--num_epochs', type=int, default=500, 
                        help="Number of training epochs.")
    parser.add_argument('--scaler_type', type=str, default='standard', choices=['standard', 'minmax', 'robust'],
                        help="Type of scaler to use for normalization.")
    parser.add_argument('--use_focal_loss', action='store_true', default=False,
                        help="Use focal loss for regression.")
    parser.add_argument('--use_gradient_clipping', action='store_true', default=True,
                        help="Use gradient clipping during training.")
    parser.add_argument('--grad_clip_value', type=float, default=0.5,
                        help="Gradient clipping value.")
    parser.add_argument('--trial_seeds', type=str, default='8,80,239,294,310',
                        help="Comma-separated list of trial seed numbers to run.")
    parser.add_argument('--output_suffix', type=str, default='robust_trials',
                        help="Suffix for the output directory name.")
    parser.add_argument('--save_model', action='store_true', default=False,
                        help="Save trained model (disabled by default for hyperparameter search).")
    
    args, _ = parser.parse_known_args()
    
    # --- Algorithm Setup ---
    if algorithms is not None:
        ALGORITHMS = [alg.strip() for alg in algorithms.split(',')]
    else:
        ALGORITHMS = [alg.strip() for alg in args.algorithms.split(',')]
    
    # --- Hyperparameter Setup ---
    HIDDEN_DIMS = [int(x.strip()) for x in args.hidden_dims.split(',')]
    DROPOUT_RATE = args.dropout_rate
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    SCALER_TYPE = args.scaler_type
    USE_FOCAL_LOSS = args.use_focal_loss
    USE_GRADIENT_CLIPPING = args.use_gradient_clipping
    GRAD_CLIP_VALUE = args.grad_clip_value
    TRIAL_SEEDS = [int(x.strip()) for x in args.trial_seeds.split(',')]
    # Convert to split indices (subtract 1 since splits are 1-based in output but 0-based in code)
    SPLIT_INDICES = [seed - 1 for seed in TRIAL_SEEDS]
    SAVE_MODEL = args.save_model
    
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzed", "preprocessed")

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_list = load_data(DATA_DIR)
    
    # Run experiments for each algorithm
    for algorithm in ALGORITHMS:
        print(f"\n{'='*80}")
        print(f"RUNNING ALGORITHM: {algorithm.upper()}")
        print(f"{'='*80}")
        
        # --- Directory Setup for this algorithm ---
        output_base_dir = f'results/direct_regression_{holdout_type}_holdout_{algorithm}_{args.output_suffix}'
        output_dir = output_base_dir
        
        counter = 1
        while os.path.exists(output_dir):
            if counter == 1:
                output_dir = f'{output_base_dir}_2nd'
            elif counter == 2:
                output_dir = f'{output_base_dir}_3rd'
            else:
                output_dir = f'{output_base_dir}_{counter+1}th'
            counter += 1

        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
        
        # Save hyperparameters for this algorithm
        hyperparams = vars(args)
        hyperparams['current_algorithm'] = algorithm
        hyperparams_path = os.path.join(output_dir, 'hyperparameters.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        
        # Store all split results for this algorithm
        all_split_metrics = []
        all_sigma_stats = {}
    
        # Perform robust trial seeds
        #trying to run 5 seeds, based on gaming account number haha, 880239294310: 8, 80, 239, 294, 310
        print(f"\nPerforming {len(TRIAL_SEEDS)} robust trial seeds: {TRIAL_SEEDS}")
        
        for i, split_idx in enumerate(SPLIT_INDICES):
            print(f"\n{'='*60}")
            print(f"TRIAL SEED {split_idx + 1} ({i + 1}/{len(TRIAL_SEEDS)}) - Algorithm: {algorithm.upper()}")
            print(f"{'='*60}")
            
            # Create category-based split with different random seed
            print(f"\nCreating category-based train/hold-out split (seed={42 + split_idx})...")
            train_data, holdout_data, holdout_info = create_category_based_split(data_list, random_seed=42 + split_idx)
            
            # Save hold-out set information
            holdout_df = pd.DataFrame(holdout_info)
            holdout_df.to_csv(os.path.join(output_dir, f'holdout_set_info_split_{split_idx + 1}.csv'), index=False)
            
            # Create training dataset
            print("\nCreating training dataset...")
            train_dataset = RegressionDataset(train_data, fit_scalers=True, scaler_type=SCALER_TYPE)
            
            # Create hold-out dataset (using training scalers)
            print("\nCreating hold-out dataset...")
            holdout_dataset = RegressionDataset(holdout_data, scalers=train_dataset.scalers, 
                                               concentration_scaler=train_dataset.concentration_scaler, fit_scalers=False)
            
            # Model-specific training and evaluation
            if algorithm == 'mlp':
                # Create data loaders for MLP
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                holdout_loader = DataLoader(holdout_dataset, batch_size=BATCH_SIZE, shuffle=False)
                
                # Create MLP model
                print("\nCreating MLP model...")
                signal_dim = train_dataset.signal_data.shape[1]
                condition_dim = train_dataset.condition_features.shape[1]
                
                model = ConcentrationRegressor(
                    signal_dim=signal_dim,
                    condition_dim=condition_dim,
                    hidden_dims=HIDDEN_DIMS,
                    dropout_rate=DROPOUT_RATE
                )
                
                print(f"MLP model created with {sum(p.numel() for p in model.parameters())} parameters")
                
                # Train MLP model
                print(f"\nTraining MLP model for split {split_idx + 1}...")
                print(f"Configuration:")
                print(f"  - Hidden dims: {HIDDEN_DIMS}")
                print(f"  - Dropout rate: {DROPOUT_RATE}")
                print(f"  - Learning rate: {LEARNING_RATE}")
                print(f"  - Batch size: {BATCH_SIZE}")
                print(f"  - Scaler type: {SCALER_TYPE}")
                print(f"  - Focal Loss: {USE_FOCAL_LOSS}")
                print(f"  - Gradient Clipping: {USE_GRADIENT_CLIPPING}")
                
                train_losses, val_losses = train_model(
                    model, train_loader, holdout_loader,
                    num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device=device,
                    use_focal_loss=USE_FOCAL_LOSS,
                    use_gradient_clipping=USE_GRADIENT_CLIPPING, grad_clip_value=GRAD_CLIP_VALUE
                )
                
                # Plot training history
                plot_training_history(train_losses, val_losses, output_dir, split_idx)
                
                # Evaluate on training set
                print("\nEvaluating on training set...")
                train_predictions, train_ground_truth, train_filenames, train_targets, train_setups, train_nontargets = evaluate_model(
                    model, train_loader, train_dataset.concentration_scaler, device
                )
                
                # Evaluate on hold-out set
                print("\nEvaluating on hold-out set...")
                test_predictions, test_ground_truth, test_filenames, test_targets, test_setups, test_nontargets = evaluate_model(
                    model, holdout_loader, train_dataset.concentration_scaler, device
                )
                
            else:
                # Sklearn models
                print(f"\nCreating {algorithm.upper()} model...")
                
                # Prepare data for sklearn
                X_train, y_train, train_filenames, train_targets, train_setups, train_nontargets = prepare_sklearn_data(train_dataset)
                X_holdout, y_holdout, test_filenames, test_targets, test_setups, test_nontargets = prepare_sklearn_data(holdout_dataset)
                
                # Create sklearn model
                try:
                    model = create_sklearn_model(algorithm, random_state=42 + split_idx)
                    print(f"{algorithm.upper()} model created")
                except (ImportError, ValueError) as e:
                    print(f"Error creating {algorithm} model: {e}")
                    continue
                
                # Train sklearn model
                print(f"\nTraining {algorithm.upper()} model for split {split_idx + 1}...")
                train_losses, val_losses = train_sklearn_model(model, X_train, y_train, X_holdout, y_holdout)
                
                # Plot training history (simple version for sklearn)
                plot_training_history(train_losses, val_losses, output_dir, split_idx)
                
                # Evaluate on training set
                print("\nEvaluating on training set...")
                train_predictions, train_ground_truth = evaluate_sklearn_model(
                    model, X_train, y_train, train_dataset.concentration_scaler
                )
                
                # Evaluate on hold-out set
                print("\nEvaluating on hold-out set...")
                test_predictions, test_ground_truth = evaluate_sklearn_model(
                    model, X_holdout, y_holdout, train_dataset.concentration_scaler
                )
        
            # Save results, which now also calculates metrics internally
            results_df = save_results(
                train_predictions, train_ground_truth, train_filenames, train_targets, train_setups, train_nontargets,
                test_predictions, test_ground_truth, test_filenames, test_targets, test_setups, test_nontargets, 
                output_dir, split_idx
            )

            # Load the saved metrics to report them and store for summary
            metrics_path = os.path.join(output_dir, f'metrics_split_{split_idx + 1}.json')
            with open(metrics_path, 'r') as f:
                split_metrics = json.load(f)
            
            train_metrics = split_metrics['overall_train_metrics']
            test_metrics = split_metrics['overall_test_metrics']

            print("\nTraining Set Metrics:")
            for key, value in train_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            print(f"\nHold-out Set Metrics:")
            for key, value in test_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Store metrics for final summary
            all_split_metrics.append({
                'split': split_idx + 1,
                'overall_test_metrics': test_metrics,
                'metrics_file': f'metrics_split_{split_idx + 1}.json'
            })
            
            # --- Generate Plots based on Legacy Script ---
            print("\nGenerating plots with Sensor Plotting style...")
            # Create a single directory for this split's plots
            split_plot_dir = os.path.join(output_dir, f"split_{split_idx + 1}_plots")
            os.makedirs(split_plot_dir, exist_ok=True)
            
            # Preprocess data for plotting (treat non-targets as regular points)
            plot_df = results_df.copy()
            plot_df.loc[plot_df['is_nontarget'] == 1, 'is_nontarget'] = 0

            # Generate modified scatter plot
            plot_modified_scatter_plot(plot_df, split_plot_dir, shuffle_idx=split_idx + 1)
            
            # Generate error bar plots (Versions A, B, C, and D)
            plot_error_bar_version_a(plot_df, split_plot_dir, shuffle_idx=split_idx + 1)
            plot_error_bar_version_b(plot_df, split_plot_dir, shuffle_idx=split_idx + 1)
            plot_error_bar_version_c(plot_df, split_plot_dir, shuffle_idx=split_idx + 1)
            plot_error_bar_version_d(plot_df, split_plot_dir, shuffle_idx=split_idx + 1)
            
            # Generate sigma error analysis and store stats
            sigma_stats = plot_sigma_error_analysis(plot_df, split_plot_dir, shuffle_idx=split_idx + 1)
            all_sigma_stats[split_idx + 1] = sigma_stats
            
            # Generate 3-sigma plot
            plot_3sigma_error_analysis(plot_df, split_plot_dir, shuffle_idx=split_idx + 1)
            
            # Save model (only if requested)
            if SAVE_MODEL:
                if algorithm == 'mlp':
                    model_path = os.path.join(output_dir, f'regression_model_split_{split_idx + 1}.pth')
                    torch.save({
                        'model_type': 'MLP',
                        'model_state_dict': model.state_dict(),
                        'signal_dim': signal_dim,
                        'condition_dim': condition_dim,
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'scalers': train_dataset.scalers,
                        'concentration_scaler': train_dataset.concentration_scaler
                    }, model_path)
                    print(f"MLP model saved to: {model_path}")
                else:
                    # For sklearn models, use pickle
                    import pickle
                    model_path = os.path.join(output_dir, f'regression_model_split_{split_idx + 1}.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump({
                            'model_type': algorithm,
                            'model': model,
                            'train_metrics': train_metrics,
                            'test_metrics': test_metrics,
                            'scalers': train_dataset.scalers,
                            'concentration_scaler': train_dataset.concentration_scaler
                        }, f)
                    print(f"{algorithm.upper()} model saved to: {model_path}")
            else:
                print("Model saving disabled for hyperparameter search")
    
        # Calculate average metrics across all splits for this algorithm
        avg_metrics = {}
        std_metrics = {}
        
        metric_keys = ['r2_score', 'mse', 'rmse', 'mae', 'pearson_corr']
        for key in metric_keys:
            values = [m['overall_test_metrics'][key] for m in all_split_metrics]
            avg_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
        
        print(f"\n\n{'='*60}")
        print(f"SUMMARY: Average Hold-out Metrics Across {len(TRIAL_SEEDS)} Robust Trial Seeds - {algorithm.upper()}")
        print(f"{'='*60}")
        for key in metric_keys:
            print(f"{key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
        
        # Calculate average target-specific metrics
        avg_category_metrics = defaultdict(lambda: defaultdict(list))
        categories = ['IgG-Au40', 'IgG-HRPAB', 'HRP', 'Bacteria']
        
        print(f"\n\nCategory-Specific Average Metrics:")
        for category in categories:
            print(f"\n{category}:")
            
            # Collect all metrics for this category across trial seeds
            for split_idx in SPLIT_INDICES:
                metrics_file = os.path.join(output_dir, f'metrics_split_{split_idx + 1}.json')
                with open(metrics_file, 'r') as f:
                    split_data = json.load(f)
                    if 'category_specific_metrics' in split_data and \
                       category in split_data['category_specific_metrics'] and \
                       'test' in split_data['category_specific_metrics'][category]:
                        
                        test_metrics = split_data['category_specific_metrics'][category]['test']
                        for key, value in test_metrics.items():
                            if value is not None:
                                avg_category_metrics[category][key].append(value)
            
            # Calculate averages for this category
            if avg_category_metrics[category]:
                for key, values in avg_category_metrics[category].items():
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        print(f"  test_{key}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Calculate average LDL metrics
        print(f"\n\nLDL and Concentration Extremes Average Metrics:")
        avg_ldl_metrics = defaultdict(lambda: defaultdict(list))
        categories_for_ldl = ['full', 'IgG-Au40', 'IgG-HRPAB', 'HRP', 'Bacteria']

        for split_idx in SPLIT_INDICES:
            metrics_file = os.path.join(output_dir, f'metrics_split_{split_idx + 1}.json')
            with open(metrics_file, 'r') as f:
                split_data = json.load(f)
                if 'ldl_metrics' in split_data:
                    for category, metrics in split_data['ldl_metrics'].items():
                        if category in categories_for_ldl:
                            for key, value in metrics.items():
                                if value is not None and 'error' not in str(value).lower():
                                    avg_ldl_metrics[category][key].append(value)

        final_avg_ldl = {}
        for category, keys in avg_ldl_metrics.items():
            print(f"\n{category}:")
            final_avg_ldl[category] = {}
            for key, values in keys.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    final_avg_ldl[category][f'avg_{key}'] = mean_val
                    final_avg_ldl[category][f'std_{key}'] = std_val
                    print(f"  {key}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Generate cross-split summary for sigma analysis
        if all_sigma_stats:
            print("\nGenerating cross-split sigma analysis summary...")
            generate_cross_split_summary(all_sigma_stats, output_dir)
        
        # Save summary metrics
        summary_metrics = {
            'algorithm': algorithm,
            'trial_seeds': TRIAL_SEEDS,
            'n_trials': len(TRIAL_SEEDS),
            'average_holdout_metrics': avg_metrics,
            'std_holdout_metrics': std_metrics,
            'average_category_metrics': {cat: {key: (np.mean(vals), np.std(vals)) for key, vals in data.items()} for cat, data in avg_category_metrics.items()},
            'average_ldl_metrics': final_avg_ldl,
            'all_split_metrics': all_split_metrics
        }
        
        summary_path = os.path.join(output_dir, 'summary_metrics.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_metrics, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        print(f"\nSummary metrics saved to: {summary_path}")
        
        print(f"\nDirect regression experiment with robust trial seeds {TRIAL_SEEDS} for {algorithm.upper()} completed successfully!")
    
    print(f"\n{'='*80}")
    print("ALL ALGORITHMS COMPLETED!")
    print(f"{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run machine learning model with specified holdout type and model name.')
    parser.add_argument('--holdout_type', type=str, default='category', choices=['category', 'concentration'], help='Type of holdout validation.')
    parser.add_argument('--model_name', type=str, default='mlp', choices=['mlp'], help='Name of the model to use (legacy parameter, use --algorithms instead).')
    
    # Model selection
    parser.add_argument('--algorithms', type=str, default='mlp', 
                        help="Comma-separated list of algorithms to run: mlp,rf,gbdt,svr,knr,xgboost,catboost,lightgbm")
    
    # Add some helpful examples
    parser.add_argument('--example', action='store_true', help='Show example usage and exit.')
    
    args = parser.parse_args()
    
    if args.example:
        print("Example usage:")
        print("1. Run only MLP:")
        print("   python No.2_direct_regression_split_igg.py --algorithms mlp")
        print("2. Run multiple algorithms:")
        print("   python No.2_direct_regression_split_igg.py --algorithms mlp,rf,xgboost")
        print("3. Run all available algorithms:")
        print("   python No.2_direct_regression_split_igg.py --algorithms mlp,rf,gbdt,svr,knr,xgboost,catboost,lightgbm")
        print("4. Custom trial seeds:")
        print("   python No.2_direct_regression_split_igg.py --algorithms rf --trial_seeds 1,5,10")
        exit()
    
    main(args.holdout_type, args.model_name, args.algorithms)