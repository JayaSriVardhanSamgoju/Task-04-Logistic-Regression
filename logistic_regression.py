import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Set styling for plots to look professional and premium
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.figsize': (8, 6),
    'axes.edgecolor': '#cccccc',
    'grid.color': '#f0f0f0'
})
PRIMARY_COLOR = '#1e3d59'  # Premium Slate Blue
SECONDARY_COLOR = '#ff6e40'  # Modern Coral
ACCENT_COLOR = '#ffc13b'  # Gold
PALETTE = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR]

def setup_directories():
    """Create project directories if they do not exist."""
    dirs = ['images', 'models', 'outputs']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

def load_and_preprocess_data(filepath):
    """Load the dataset, handle missing values, clean up column names, and encode targets."""
    print("--- 1. Loading and Preprocessing Data ---")
    df = pd.read_csv(filepath)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Drop completely empty columns (e.g. due to trailing comma)
    df = df.dropna(axis=1, how='all')
    # Also drop unnamed columns if any
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        
    print(f"Dataset shape: {df.shape}")
    
    # Check for target column
    if 'diagnosis' not in df.columns:
        raise ValueError("Target column 'diagnosis' not found in dataset.")
        
    # Map target: M -> 1 (Malignant), B -> 0 (Benign)
    df['diagnosis_encoded'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Drop features that are not predictors
    # 'id' is a key, not a predictor. 'diagnosis' is the raw target.
    # 'diagnosis_encoded' is the target to be predicted.
    cols_to_drop = ['id', 'diagnosis', 'diagnosis_encoded']
    features_df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Impute missing values (if any) with the median
    if features_df.isnull().sum().sum() > 0:
        print("Handling missing values using median imputation...")
        features_df = features_df.fillna(features_df.median())
        
    print("Preprocessing completed.")
    return df, features_df

def generate_eda_plots(df):
    """Generate exploratory plots before modeling."""
    print("--- 2. Generating EDA Plots ---")
    
    # 1. Dataset Preview (Table view as PNG)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    # Clean up display columns for preview
    preview_df = df.iloc[:8, :7].copy()
    table = ax.table(
        cellText=preview_df.values, 
        colLabels=preview_df.columns, 
        cellLoc='center', 
        loc='center',
        cellColours=[['#f8f9fa']*len(preview_df.columns)]*len(preview_df)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    # Style header row
    for j in range(len(preview_df.columns)):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor(PRIMARY_COLOR)
    plt.title("Dataset Preview (First 8 Rows, First 7 Columns)", fontsize=14, weight='bold', pad=15, color=PRIMARY_COLOR)
    plt.tight_layout()
    plt.savefig('images/dataset_preview.png', dpi=300)
    plt.close()
    print("Saved images/dataset_preview.png")

    # 2. Class Distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df['diagnosis'].value_counts()
    bars = ax.bar(counts.index, counts.values, color=[SECONDARY_COLOR, PRIMARY_COLOR], width=0.5, edgecolor='#333333', linewidth=1)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.0, 
            height + 10, 
            f'{height} ({height/len(df)*100:.1f}%)', 
            ha='center', 
            va='bottom', 
            fontsize=11, 
            weight='bold'
        )
        
    ax.set_title("Target Class Distribution (Diagnosis)", fontsize=14, weight='bold', pad=15, color=PRIMARY_COLOR)
    ax.set_xlabel("Diagnosis (M = Malignant, B = Benign)", fontsize=12, labelpad=10)
    ax.set_ylabel("Count", fontsize=12, labelpad=10)
    ax.set_ylim(0, max(counts.values) * 1.15)
    sns.despine()
    plt.tight_layout()
    plt.savefig('images/class_distribution.png', dpi=300)
    plt.close()
    print("Saved images/class_distribution.png")

    # 3. Correlation Heatmap (using mean columns to prevent clutter)
    mean_cols = [col for col in df.columns if '_mean' in col] + ['diagnosis_encoded']
    corr_matrix = df[mean_cols].corr()
    
    # Clean label names for the heatmap
    clean_labels = [col.replace('_mean', '').title() for col in mean_cols]
    clean_labels = [col.replace('Encoded', 'Diagnosis') for col in clean_labels]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=0.5, 
        ax=ax,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title("Correlation Heatmap of Mean Features & Diagnosis", fontsize=14, weight='bold', pad=15, color=PRIMARY_COLOR)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('images/correlation_heatmap.png', dpi=300)
    plt.close()
    print("Saved images/correlation_heatmap.png")

def generate_sigmoid_plot():
    """Generate a clean explanation plot of the Sigmoid activation function."""
    print("--- 3. Generating Sigmoid Function Plot ---")
    z = np.linspace(-10, 10, 200)
    s = 1 / (1 + np.exp(-z))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z, s, color=PRIMARY_COLOR, linewidth=2.5, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')
    
    # Decision threshold lines
    ax.axhline(0.5, color=SECONDARY_COLOR, linestyle='--', linewidth=1.5, label='Decision Threshold (0.5)')
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    
    # Fill positive/negative class areas
    ax.fill_between(z, 0.5, s, where=(z >= 0), color='#e3f2fd', alpha=0.5, label='Predict Class 1 (Malignant)')
    ax.fill_between(z, 0.5, s, where=(z < 0), color='#ffebee', alpha=0.5, label='Predict Class 0 (Benign)')
    
    ax.set_title("Sigmoid Activation Function in Logistic Regression", fontsize=14, weight='bold', pad=15, color=PRIMARY_COLOR)
    ax.set_xlabel("Linear Logit $z = w^T x + b$", fontsize=12, labelpad=10)
    ax.set_ylabel("Probability $\sigma(z)$", fontsize=12, labelpad=10)
    ax.legend(loc='lower right', frameon=True, facecolor='white')
    ax.set_ylim(-0.05, 1.05)
    sns.despine()
    plt.tight_layout()
    plt.savefig('images/sigmoid_function.png', dpi=300)
    plt.close()
    print("Saved images/sigmoid_function.png")

def train_pipeline(features_df, target_series):
    """Split data, build, and train the logistic regression pipeline."""
    print("--- 4. Splitting Data and Training Model ---")
    # Split into Train and Test (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, 
        target_series, 
        test_size=0.2, 
        random_state=42, 
        stratify=target_series
    )
    
    # Define sklearn Pipeline with StandardScaler and LogisticRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            penalty='l2', 
            C=1.0, 
            random_state=42, 
            max_iter=10000
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    print("Pipeline trained successfully (StandardScaler + LogisticRegression).")
    return pipeline, X_train, X_test, y_train, y_test

def evaluate_and_plot_results(pipeline, X_test, y_test, X_train):
    """Run model predictions and save evaluation metrics and plots."""
    print("--- 5. Evaluating Model ---")
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=False, 
        ax=ax,
        xticklabels=['Benign (0)', 'Malignant (1)'],
        yticklabels=['Benign (0)', 'Malignant (1)'],
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    ax.set_title("Confusion Matrix", fontsize=14, weight='bold', pad=15, color=PRIMARY_COLOR)
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Class", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved images/confusion_matrix.png")

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PRIMARY_COLOR, linewidth=2.5, label=f'Logistic Regression (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Chance')
    
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, weight='bold', pad=15, color=PRIMARY_COLOR)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, labelpad=10)
    ax.legend(loc='lower right', frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig('images/roc_curve.png', dpi=300)
    plt.close()
    print("Saved images/roc_curve.png")

    # 3. Precision-Recall Curve & Threshold Analysis
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=SECONDARY_COLOR, linewidth=2.5, label='Precision-Recall Curve')
    ax.set_title("Precision-Recall Curve", fontsize=14, weight='bold', pad=15, color=PRIMARY_COLOR)
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12, labelpad=10)
    ax.set_ylabel("Precision (PPV)", fontsize=12, labelpad=10)
    ax.legend(loc='lower left', frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig('images/precision_recall_curve.png', dpi=300)
    plt.close()
    print("Saved images/precision_recall_curve.png")

    # 4. Feature Importance Plot (Standardized Coefficients)
    model = pipeline.named_steps['classifier']
    coefficients = model.coef_[0]
    feature_names = X_train.columns
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values(by='Abs_Coefficient', ascending=False)
    
    # Plot top 15 features to avoid a messy plot
    top_n = min(15, len(coef_df))
    plot_df = coef_df.head(top_n).sort_values(by='Coefficient', ascending=True)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [PRIMARY_COLOR if val > 0 else SECONDARY_COLOR for val in plot_df['Coefficient']]
    ax.barh(plot_df['Feature'], plot_df['Coefficient'], color=colors, edgecolor='#555555', height=0.6)
    
    # Highlight interpretations
    ax.set_title(f"Top {top_n} Standardized Coefficients (Feature Importance)", fontsize=14, weight='bold', pad=15, color=PRIMARY_COLOR)
    ax.set_xlabel("Coefficient Weight\n(Positive: Malignant risk factor | Negative: Benign indicator)", fontsize=12, labelpad=10)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('images/feature_importance.png', dpi=300)
    plt.close()
    print("Saved images/feature_importance.png")

    # Save outputs
    # Write classification report to file
    report = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    with open('outputs/classification_report.txt', 'w') as f:
        f.write("=== LOGISTIC REGRESSION CLASSIFICATION REPORT ===\n\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n\n")
        f.write("Detailed Report:\n")
        f.write(report)
    print("Saved outputs/classification_report.txt")

    # Save predictions.csv
    predictions_df = X_test.copy()
    predictions_df['Actual_Diagnosis'] = y_test
    predictions_df['Predicted_Diagnosis'] = y_pred
    predictions_df['Malignant_Probability'] = y_prob
    predictions_df.to_csv('outputs/predictions.csv', index=False)
    print("Saved outputs/predictions.csv")

    # Display console summary
    print("\n--- MODEL PERFORMANCE RESULTS SUMMARY ---")
    print(f"Accuracy:  {acc:.4%}")
    print(f"Precision: {prec:.4%}")
    print(f"Recall:    {rec:.4%}")
    print(f"F1-Score:  {f1:.4%}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("----------------------------------------\n")

def save_model(pipeline):
    """Serialize the trained pipeline to file."""
    print("--- 6. Saving Model Pipeline ---")
    joblib.dump(pipeline, 'models/logistic_regression_model.pkl')
    print("Saved models/logistic_regression_model.pkl")

def main():
    setup_directories()
    
    dataset_path = 'dataset/data.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset at {dataset_path}. Please place the file and run again.")
        
    df, features_df = load_and_preprocess_data(dataset_path)
    
    # Target column is the encoded diagnosis
    target_series = df['diagnosis_encoded']
    
    generate_eda_plots(df)
    generate_sigmoid_plot()
    
    pipeline, X_train, X_test, y_train, y_test = train_pipeline(features_df, target_series)
    
    evaluate_and_plot_results(pipeline, X_test, y_test, X_train)
    
    save_model(pipeline)
    print("Pipeline executed successfully!")

if __name__ == '__main__':
    main()
