import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_metrics(y_true, y_pred):
    """Evaluates classification metrics including standard and rejection metrics."""
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Rejection Stats (predictions equal to -1)
    valid_mask = (y_pred != -1)
    rejection_rate = float(np.mean(~valid_mask))
    
    # Overall Accuracy (treating -1 as incorrect)
    overall_acc = float(accuracy_score(y_true, y_pred))
    
    # Filtered Accuracy (only on valid predictions)
    if np.sum(valid_mask) > 0:
        filtered_acc = float(accuracy_score(y_true[valid_mask], y_pred[valid_mask]))
    else:
        filtered_acc = 0.0
        
    # Standard metrics on the raw predictions (weighted across classes)
    # Note: scikit-learn metrics might complain about -1 class, we compute them treating -1 as a distinct class or on valid mask
    # To be fair to the metric evaluation, we treat -1 as incorrect (which matches normal accuracy calculation)
    # We map -1 to a dummy class index outside standard classes if it exists, or just compute on all
    # For Precision/Recall/F1, we compute on all classes (including -1 if present) to match sklearn standard behavior
    prec = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    rec = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    return {
        'Accuracy': overall_acc, 
        'Precision': prec, 
        'Recall': rec, 
        'F1-score': f1,
        'Rejection Rate': rejection_rate,
        'Filtered Accuracy': filtered_acc
    }

def evaluate_inference_speed(model, X_test):
    """Evaluates inference speed (ms per sample and FPS)."""
    start_time = time.time()
    _ = model.predict(X_test)
    end_time = time.time()
    
    total_time_ms = (end_time - start_time) * 1000
    ms_per_sample = total_time_ms / len(X_test)
    fps = 1000 / ms_per_sample if ms_per_sample > 0 else float('inf')
    
    return {
        'Total Time (ms)': float(total_time_ms), 
        'ms/sample': float(ms_per_sample), 
        'FPS': float(fps)
    }

def add_gaussian_noise(X, std=0.1):
    """Adds Gaussian noise to features."""
    noise = np.random.normal(0.0, std, X.shape)
    return X + noise

def add_salt_and_pepper_noise(X, prob=0.05):
    """Adds Salt and Pepper noise to features."""
    noisy_X = np.copy(X)
    # Salt (set to max value in X)
    num_salt = np.ceil(prob * 0.5 * X.size)
    coords_salt = [np.random.randint(0, i, int(num_salt)) for i in X.shape]
    noisy_X[tuple(coords_salt)] = np.max(X) if np.max(X) > 0 else 1.0
    
    # Pepper (set to min value in X)
    num_pepper = np.ceil(prob * 0.5 * X.size)
    coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in X.shape]
    noisy_X[tuple(coords_pepper)] = np.min(X) if np.min(X) < 0 else 0.0
    return noisy_X

def add_feature_dropout_noise(X, prob=0.1):
    """Randomly zeros out features (masking noise)."""
    mask = np.random.binomial(1, 1 - prob, X.shape)
    return X * mask

def evaluate_robustness_gradients(model, X_test, y_test):
    """Evaluates model accuracy across noise intensity gradients."""
    y_test = np.array(y_test)
    
    # Define intensity gradients
    gaussian_stds = [0.01, 0.05, 0.1, 0.2, 0.5]
    sp_probs = [0.01, 0.05, 0.1, 0.2, 0.3]
    dropout_probs = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    results = {
        'gaussian': [],
        'salt_pepper': [],
        'feature_dropout': []
    }
    
    # Evaluate Gaussian
    for std in gaussian_stds:
        X_noisy = add_gaussian_noise(X_test, std=std)
        preds = model.predict(X_noisy, raw=True)
        acc = float(accuracy_score(y_test, preds))
        results['gaussian'].append({'intensity': std, 'accuracy': acc})
        
    # Evaluate Salt & Pepper
    for prob in sp_probs:
        X_noisy = add_salt_and_pepper_noise(X_test, prob=prob)
        preds = model.predict(X_noisy, raw=True)
        acc = float(accuracy_score(y_test, preds))
        results['salt_pepper'].append({'intensity': prob, 'accuracy': acc})
        
    # Evaluate Feature Dropout
    for prob in dropout_probs:
        X_noisy = add_feature_dropout_noise(X_test, prob=prob)
        preds = model.predict(X_noisy, raw=True)
        acc = float(accuracy_score(y_test, preds))
        results['feature_dropout'].append({'intensity': prob, 'accuracy': acc})
        
    return results

def get_confusion_matrix_data(y_true, y_pred, num_classes):
    """Computes standard confusion matrix."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # We only compute confusion matrix for valid predictions (filter out -1)
    # Or keep -1 as a separate column? To keep it standard and matching class names,
    # we filter out -1 predictions so it matches standard class size.
    valid_mask = (y_pred != -1)
    
    if np.sum(valid_mask) > 0:
        cm = confusion_matrix(y_true[valid_mask], y_pred[valid_mask], labels=range(num_classes))
    else:
        cm = np.zeros((num_classes, num_classes), dtype=int)
        
    return cm.tolist()

def evaluate_overfitting(model, X_train, y_train, X_test, y_test):
    """Compares train accuracy vs test accuracy with absolute and relative differences."""
    train_pred = model.predict(X_train, raw=True)
    test_pred = model.predict(X_test, raw=True)
    
    train_acc = float(accuracy_score(y_train, train_pred))
    test_acc = float(accuracy_score(y_test, test_pred))
    
    diff_abs = train_acc - test_acc
    diff_rel = (diff_abs / train_acc) if train_acc > 0 else 0.0
    
    return {
        'Train Acc': train_acc,
        'Test Acc': test_acc,
        'Absolute Diff': float(diff_abs),
        'Relative Diff': float(diff_rel),
        'Overfitting Risk': 'High' if diff_abs > 0.08 else 'Low'
    }

if __name__ == "__main__":
    from core.models import SVMModel
    
    print("Testing evaluator components...")
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 3, 100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 3, 20)
    
    svm = SVMModel()
    svm.train(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    
    print("\nMetrics:")
    print(evaluate_metrics(y_test, y_pred))
    
    print("\nSpeed:")
    print(evaluate_inference_speed(svm, X_test))
    
    print("\nOverfitting:")
    print(evaluate_overfitting(svm, X_train, y_train, X_test, y_test))
    
    print("\nNoise Gradients:")
    print(evaluate_robustness_gradients(svm, X_test, y_test))
    
    print("\nConfusion Matrix:")
    print(get_confusion_matrix_data(y_test, y_pred, num_classes=3))
    
    print("\nEvaluator test passed successfully.")
