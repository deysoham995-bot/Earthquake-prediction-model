import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive evaluation and visualization of model performance"""
    
    def __init__(self, y_true):
        self.y_true = y_true
        self.predictions = {}
        self.metrics = {}
        
    def add_prediction(self, model_name, y_pred):
        """Add model predictions"""
        self.predictions[model_name] = y_pred
        
    def evaluate_all(self):
        """Calculate all metrics for all models"""
        for model_name, y_pred in self.predictions.items():
            self.metrics[model_name] = self._calculate_metrics(y_pred)
        
        return self.get_metrics_dataframe()
    
    def _calculate_metrics(self, y_pred):
        """Calculate comprehensive metrics"""
        mse = mean_squared_error(self.y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_true, y_pred)
        median_ae = median_absolute_error(self.y_true, y_pred)
        r2 = r2_score(self.y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error) - robust metric
        mape = np.mean(np.abs((self.y_true - y_pred) / self.y_true)) * 100
        
        # Correlation coefficient
        correlation = np.corrcoef(self.y_true, y_pred)[0, 1]
        
        # Residuals statistics
        residuals = self.y_true - y_pred
        residuals_std = np.std(residuals)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Median AE': median_ae,
            'R² Score': r2,
            'MAPE (%)': mape,
            'Correlation': correlation,
            'Residuals Std': residuals_std
        }
    
    def get_metrics_dataframe(self):
        """Return metrics as DataFrame"""
        return pd.DataFrame(self.metrics).T
    
    def plot_actual_vs_predicted(self, output_path='models/evaluation/'):
        """Plot actual vs predicted for all models"""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        n_models = len(self.predictions)
        grid_rows = (n_models + 1) // 2
        grid_cols = min(2, n_models)
        
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(14, 6*grid_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        
        for idx, (model_name, y_pred) in enumerate(self.predictions.items()):
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(self.y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(self.y_true.min(), y_pred.min())
            max_val = max(self.y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            
            # Labels and title
            r2 = self.metrics[model_name]['R² Score']
            rmse = self.metrics[model_name]['RMSE']
            ax.set_xlabel('Actual Magnitude', fontsize=11, fontweight='bold')
            ax.set_ylabel('Predicted Magnitude', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name}\nR²={r2:.4f}, RMSE={rmse:.4f}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for idx in range(len(self.predictions), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{output_path}actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}actual_vs_predicted.png")
        plt.close()
    
    def plot_residuals(self, output_path='models/evaluation/'):
        """Plot residuals distribution for all models"""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        n_models = len(self.predictions)
        grid_rows = (n_models + 1) // 2
        grid_cols = min(2, n_models)
        
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(14, 6*grid_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        
        for idx, (model_name, y_pred) in enumerate(self.predictions.items()):
            ax = axes[idx]
            residuals = self.y_true - y_pred
            
            # Histogram with KDE
            ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
            
            # Add normal distribution curve
            mu, sigma = residuals.mean(), residuals.std()
            x = np.linspace(residuals.min(), residuals.max(), 100)
            ax.plot(x, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2)), 'r-', lw=2)
            
            ax.axvline(0, color='g', linestyle='--', lw=2, label='Zero error')
            ax.set_xlabel('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Density', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name} - Residuals Distribution', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for idx in range(len(self.predictions), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{output_path}residuals.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}residuals.png")
        plt.close()
    
    def plot_error_distribution(self, output_path='models/evaluation/'):
        """Plot absolute error distribution"""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for model_name, y_pred in self.predictions.items():
            errors = np.abs(self.y_true - y_pred)
            ax.hist(errors, bins=50, alpha=0.5, label=model_name, edgecolor='black')
        
        ax.set_xlabel('Absolute Error (Magnitude)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Absolute Error Distribution by Model', fontweight='bold', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_path}error_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}error_distribution.png")
        plt.close()
    
    def plot_metrics_comparison(self, output_path='models/evaluation/'):
        """Plot comparison of key metrics across models"""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        metrics_df = self.get_metrics_dataframe()
        
        # Select key metrics for visualization
        key_metrics = ['RMSE', 'MAE', 'R² Score', 'MAPE (%)']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(key_metrics):
            ax = axes[idx]
            values = metrics_df[metric]
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
            
            bars = ax.bar(values.index, values.values, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_path}metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}metrics_comparison.png")
        plt.close()
    
    def generate_report(self, output_path='models/evaluation/'):
        """Generate comprehensive evaluation report"""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        metrics_df = self.get_metrics_dataframe()
        
        # Save metrics table
        metrics_df.to_csv(f'{output_path}detailed_metrics.csv')
        print(f"✓ Saved: {output_path}detailed_metrics.csv")
        
        # Generate text report
        report = "="*80 + "\n"
        report += "EARTHQUAKE MAGNITUDE PREDICTION - COMPREHENSIVE EVALUATION REPORT\n"
        report += "="*80 + "\n\n"
        
        report += "METRICS SUMMARY:\n"
        report += "-"*80 + "\n"
        report += metrics_df.to_string() + "\n\n"
        
        # Model rankings
        report += "MODEL RANKINGS:\n"
        report += "-"*80 + "\n"
        report += "By R² Score (higher is better):\n"
        r2_ranking = metrics_df['R² Score'].sort_values(ascending=False)
        for rank, (model, score) in enumerate(r2_ranking.items(), 1):
            report += f"  {rank}. {model}: {score:.6f}\n"
        
        report += "\nBy RMSE (lower is better):\n"
        rmse_ranking = metrics_df['RMSE'].sort_values(ascending=True)
        for rank, (model, score) in enumerate(rmse_ranking.items(), 1):
            report += f"  {rank}. {model}: {score:.6f}\n"
        
        report += "\nBy MAE (lower is better):\n"
        mae_ranking = metrics_df['MAE'].sort_values(ascending=True)
        for rank, (model, score) in enumerate(mae_ranking.items(), 1):
            report += f"  {rank}. {model}: {score:.6f}\n"
        
        # Save report
        with open(f'{output_path}evaluation_report.txt', 'w') as f:
            f.write(report)
        
        print(f"✓ Saved: {output_path}evaluation_report.txt")
        print("\n" + "="*80)
        print(report)
