#!/usr/bin/env python3
"""
Generate CPICANN Results Visualization
Creates performance comparison plots for the CPICANN models
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_cpicann_performance_plot():
    """Create performance comparison plot for CPICANN models"""
    
    # Data from results
    sample_sizes = [100, 1000, 10000, 50000, 100000]
    cpicann_single = [91.00, 87.80, 86.76, 87.25, 87.40]
    attention_only = [82.00, 82.60, 83.68, 84.29, None]
    cnn_only = [78.00, 78.30, 79.09, 79.62, None]
    cpicann_bi = [65.00, 65.10, 67.05, 67.95, None]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot lines
    plt.plot(sample_sizes, cpicann_single, 'o-', 
             label='CPICANN Single-Phase', linewidth=2, markersize=8, color='#1f77b4')
    plt.plot(sample_sizes[:4], attention_only[:4], 's-', 
             label='Attention-Only', linewidth=2, markersize=8, color='#ff7f0e')
    plt.plot(sample_sizes[:4], cnn_only[:4], '^-', 
             label='CNN-Only', linewidth=2, markersize=8, color='#2ca02c')
    plt.plot(sample_sizes[:4], cpicann_bi[:4], 'd-', 
             label='CPICANN Bi-Phase', linewidth=2, markersize=8, color='#d62728')
    
    # Add horizontal line for original paper performance
    plt.axhline(y=87.5, color='red', linestyle='--', alpha=0.7, 
                label='Original CPICANN (87.5%)', linewidth=2)
    
    # Customize plot
    plt.xlabel('Test Sample Size', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('CPICANN Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.ylim(60, 95)
    
    # Add annotations
    plt.annotate('Best Performance', xy=(100000, 87.4), xytext=(50000, 90),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cpicann_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Performance comparison plot saved as 'cpicann_performance_comparison.png'")

def create_performance_table():
    """Create performance summary table"""
    
    data = {
        'Model Architecture': [
            'CPICANN Single-Phase',
            'Attention-Only', 
            'CNN-Only',
            'CPICANN Bi-Phase',
            'Original CPICANN'
        ],
        '100 samples': [91.00, 82.00, 78.00, 65.00, '-'],
        '1,000 samples': [87.80, 82.60, 78.30, 65.10, '-'],
        '10,000 samples': [86.76, 83.68, 79.09, 67.05, '-'],
        '50,000 samples': [87.25, 84.29, 79.62, 67.95, '-'],
        '100,000 samples': [87.40, '-', '-', '-', '87.5%']
    }
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    df.to_csv('cpicann_performance_table.csv', index=False)
    
    print("✅ Performance table saved as 'cpicann_performance_table.csv'")
    print("\nPerformance Summary Table:")
    print(df.to_string(index=False))
    
    return df

def create_architecture_comparison_plot():
    """Create bar chart comparing architectures at 50,000 samples"""
    
    # Shorter labels to prevent overlapping
    architectures = ['CPICANN\nSingle-Phase', 'Attention-\nOnly', 'CNN-\nOnly', 'CPICANN\nBi-Phase\n(3 Phases)', 'CPICANN\nBi-Phase\n(2 Phases)']
    accuracies = [87.25, 84.29, 79.62, 67.95, 51.20]
    colors = ['#AEC6CF',  # pastel blue
              '#FFD1B3',  # pastel orange  
              '#B5EAD7',  # pastel green
              '#FFB7B2',  # pastel red/pink
              '#D7BDE2']  # pastel purple
    
    # Set font style to match the training losses plot
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    
    plt.figure(figsize=(12, 8))  # Wider figure for better spacing
    bars = plt.bar(architectures, accuracies, color=colors, alpha=1.0, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add horizontal line for original paper
    plt.axhline(y=87.5, color='red', linestyle='--', alpha=1.0, 
                label='Original CPICANN (87.5%)', linewidth=2)
    
    plt.ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('CPICANN Architecture Comparison on Validation Data', fontsize=16, fontweight='bold')
    plt.ylim(45, 92)  # Lower y-limit to show the 51.2% bar better
    # Add dotted grid lines to match the training losses plot style
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    plt.legend(fontsize=12, loc='upper right')
    
    # Rotate x-axis labels to prevent overlapping
    # Rotate x-axis labels to prevent overlapping
    plt.xticks(rotation=0, ha='center', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('cpicann_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Architecture comparison plot saved as 'cpicann_architecture_comparison.png'")

if __name__ == "__main__":
    print("🚀 Generating CPICANN Results Visualizations")
    print("=" * 50)
    
    # Create plots
    create_cpicann_performance_plot()
    create_architecture_comparison_plot()
    
    # Create table
    df = create_performance_table()
    
    print("\n✅ All visualizations and tables generated successfully!")
    print("\nFiles created:")
    print("- cpicann_performance_comparison.png")
    print("- cpicann_architecture_comparison.png") 
    print("- cpicann_performance_table.csv")
