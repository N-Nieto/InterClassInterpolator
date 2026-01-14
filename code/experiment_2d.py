# %%
import sys
import os

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from lib.data_simulation import DataSimulator
from SMOTE_to_balance_class_per_site.code.lib.ICI_correction import SMOTECorrector
from lib.utils import ModelEvaluator, ResultAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np


def main():
    """Run the 2D multi-site data experiment with configurable parameters"""
    
    # =========================================================================
    # CONFIGURATION PARAMETERS - Modify these for different experiments
    # =========================================================================
    
    # Experiment setup
    RANDOM_STATE = 42
    N_SAMPLES = 1000
    N_FEATURES = 2  # Must be 2 for 2D visualization
    N_SITES = 2
    BALANCE_PER_SITE = [0.8, 0.2]  # Class balance for each site
    
    # Advanced data simulation parameters
    SITE_EFFECT_STRENGTH = 3.1
    
    # Class distributions (same for all sites)
    # Format: [[mean_x, mean_y] for class 0], [[mean_x, mean_y] for class 1]]
    CLASS_MEANS = [[0, 0], [2, 2]]    # Class 0 centered at (0,0), Class 1 at (2,2)
    CLASS_STDS = [[1, 1.5], [1.5, 1]] # Different variances for each class/feature
    
    # Model parameters
    RF_N_ESTIMATORS = 100
    LR_MAX_ITER = 1000
    
    # Cross-validation parameters
    CV_N_SPLITS = 10
    CV_N_REPEATS = 10
    
    # SMOTE parameters
    SMOTE_K_NEIGHBORS = 5
    REMOVE_SITE_EFFECT = True
    
    # File paths
    DATA_DIR = '../data'
    OUTPUT_DIR = '../outputs'
    DATA_FILENAME = 'simulated_data_2d_advanced.csv'
    DECISION_BOUNDARY_PLOT = 'decision_boundaries_advanced_2d.png'
    COMPARISON_PLOT = 'comparison_advanced_2d.png'
    REPORT_FILENAME = 'report_advanced_2d.txt'
    
    # =========================================================================
    # EXPERIMENT SETUP - Don't modify below unless necessary
    # =========================================================================
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize components with configured parameters
    data_simulator = DataSimulator(random_state=RANDOM_STATE)
    smote_corrector = SMOTECorrector(random_state=RANDOM_STATE)
    model_evaluator = ModelEvaluator(random_state=RANDOM_STATE)
    result_analyzer = ResultAnalyzer()
    
    # Define models with configured parameters
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, 
            random_state=RANDOM_STATE
        ),
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE, 
            max_iter=LR_MAX_ITER
        )
    }
    
    scaler = StandardScaler()
    results = {}
    
    print("=== ADVANCED 2D MULTI-SITE DATA SIMULATION EXPERIMENT ===")
    print("Configuration:")
    print(f"  Samples: {N_SAMPLES}, Features: {N_FEATURES}, Sites: {N_SITES}")
    print(f"  Site balance: {BALANCE_PER_SITE}")
    print(f"  Site effect strength: {SITE_EFFECT_STRENGTH}")
    print(f"  Class 0 means: {CLASS_MEANS[0]}, stds: {CLASS_STDS[0]}")
    print(f"  Class 1 means: {CLASS_MEANS[1]}, stds: {CLASS_STDS[1]}")
    print(f"  CV: {CV_N_REPEATS} repeats of {CV_N_SPLITS}-fold")
    print(f"  SMOTE k-neighbors: {SMOTE_K_NEIGHBORS}")
    print(f"  Remove site effect: {REMOVE_SITE_EFFECT}")
    print("=" * 70)
    
    # Generate advanced 2D data with realistic class distributions
    print("\n=== GENERATING ADVANCED 2D MULTI-SITE DATA ===")
    X_imbalanced, y_imbalanced, site_imbalanced = data_simulator.simulate_multi_site_data_2d_advanced(
        n_sites=N_SITES, 
        n_samples=N_SAMPLES, 
        balance_per_site=BALANCE_PER_SITE,
        site_effect_strength=SITE_EFFECT_STRENGTH,
        class_means=CLASS_MEANS,
        class_stds=CLASS_STDS
    )
    
    # Save the 2D data
    data_filepath = os.path.join(DATA_DIR, DATA_FILENAME)
    data_simulator.save_simulated_data(
        X_imbalanced, y_imbalanced, site_imbalanced, 
        data_filepath
    )
    
    # Get detailed statistics
    stats = data_simulator.get_data_statistics(X_imbalanced, y_imbalanced, site_imbalanced)
    print(f"Advanced 2D Data statistics:")
    print(f"  Total samples: {stats['n_samples']}")
    print(f"  Features: {stats['n_features']}")
    print(f"  Sites: {stats['n_sites']}")
    print(f"  Overall class balance: {stats['overall_class_balance']:.3f}")
    print(f"  Site-target correlation: {stats['site_target_correlation']:.3f}")
    
    print(f"\n  Class statistics:")
    for class_label in [0, 1]:
        class_stats = stats['class_statistics'][f'class_{class_label}']
        print(f"    Class {class_label}:")
        print(f"      Samples: {class_stats['n_samples']}")
        print(f"      Feature means: {[f'{m:.3f}' for m in class_stats['feature_means']]}")
        print(f"      Feature stds: {[f'{s:.3f}' for s in class_stats['feature_stds']]}")
    
    print(f"\n  Site statistics:")
    for site in range(N_SITES):
        site_stats = stats['site_statistics'][f'site_{site}']
        print(f"    Site {site}:")
        print(f"      Samples: {site_stats['n_samples']}")
        print(f"      Class balance: {site_stats['class_balance']:.3f}")
        print(f"      Class 0: {site_stats['n_class_0']}, Class 1: {site_stats['n_class_1']}")
    
    # Create 2D visualization for imbalanced data
    print("\n=== CREATING 2D VISUALIZATIONS ===")
    decision_boundary_path = os.path.join(OUTPUT_DIR, DECISION_BOUNDARY_PLOT)
    result_analyzer.plot_decision_boundaries_2d(
        X_imbalanced, y_imbalanced, site_imbalanced, models,
        f"Advanced Data (Class 0: {CLASS_MEANS[0]}, Class 1: {CLASS_MEANS[1]})",
        decision_boundary_path
    )
    
    # Generate balanced data for comparison (using same class distributions)
    X_balanced, y_balanced, site_balanced = data_simulator.simulate_multi_site_data_2d_advanced(
        n_sites=N_SITES, 
        n_samples=N_SAMPLES, 
        balance_per_site=[0.5, 0.5],  # Force balanced for comparison
        site_effect_strength=SITE_EFFECT_STRENGTH,
        class_means=CLASS_MEANS,
        class_stds=CLASS_STDS,
    )
    
    # Apply SMOTE to training data and visualize
    print("\n=== APPLYING SMOTE AND VISUALIZING ===")
    
    # For visualization, let's apply SMOTE to the entire dataset
    X_smote, y_smote, site_smote = smote_corrector.apply_smote_per_site_to_training(
        X_imbalanced, y_imbalanced, site_imbalanced, 
        k_neighbors=SMOTE_K_NEIGHBORS, 
        remove_site_effect=REMOVE_SITE_EFFECT
    )
    
    # Create comparison across scenarios
    scenarios = {
        'Imbalanced': (X_imbalanced, y_imbalanced, site_imbalanced),
        'Balanced': (X_balanced, y_balanced, site_balanced),
        'SMOTE_Corrected': (X_smote, y_smote, site_smote)
    }
    
    comparison_path = os.path.join(OUTPUT_DIR, COMPARISON_PLOT)
    result_analyzer.plot_comparison_2d(scenarios, models, comparison_path)

    # Run evaluations
    print("\n=== RUNNING MODEL EVALUATIONS ===")
    
    # Scenario 1: Imbalanced data
    print("\n=== SCENARIO 1: Imbalanced Data ===")
    scenario1_results = {}
    for name, model in models.items():
        scores = model_evaluator.evaluate_model_with_smote(
            X_imbalanced, y_imbalanced, site_imbalanced, model,
            scaler if 'Logistic' in name else None, 
            use_smote=False,
            k_neighbors=SMOTE_K_NEIGHBORS,
            smote_corrector=smote_corrector
        )
        scenario1_results[name] = scores
        mean_auc = np.mean(scores)
        std_auc = np.std(scores)
        print(f"{name}: AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    
    results['Imbalanced'] = scenario1_results
    
    # Scenario 2: Balanced data
    print("\n=== SCENARIO 2: Balanced Data ===")
    scenario2_results = {}
    for name, model in models.items():
        scores = model_evaluator.evaluate_model_with_smote(
            X_balanced, y_balanced, site_balanced, model,
            scaler if 'Logistic' in name else None, 
            use_smote=False,
            k_neighbors=SMOTE_K_NEIGHBORS,
            smote_corrector=smote_corrector
        )
        scenario2_results[name] = scores
        mean_auc = np.mean(scores)
        std_auc = np.std(scores)
        print(f"{name}: AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    
    results['Balanced'] = scenario2_results
    
    # Scenario 3: SMOTE corrected
    print("\n=== SCENARIO 3: SMOTE Corrected ===")
    scenario3_results = {}
    for name, model in models.items():
        scores = model_evaluator.evaluate_model_with_smote(
            X_imbalanced, y_imbalanced, site_imbalanced, model,
            scaler if 'Logistic' in name else None, 
            use_smote=True, 
            k_neighbors=SMOTE_K_NEIGHBORS,
            smote_corrector=smote_corrector
        )
        scenario3_results[name] = scores
        mean_auc = np.mean(scores)
        std_auc = np.std(scores)
        print(f"{name}: AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    
    results['SMOTE_Corrected'] = scenario3_results
    
    # Analyze results
    print("\n=== GENERATING RESULTS ===")
    report = result_analyzer.perform_statistical_analysis(results)
    print(report)
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, REPORT_FILENAME)
    result_analyzer.save_report(report, report_path)
    
    # Create performance plot
    performance_plot_path = os.path.join(OUTPUT_DIR, 'performance_comparison_advanced_2d.png')
    result_analyzer.create_performance_plot(results, performance_plot_path)
    
    print("\n=== ADVANCED 2D EXPERIMENT COMPLETED ===")
    print("Outputs saved to:")
    print(f"  - {decision_boundary_path}")
    print(f"  - {comparison_path}")
    print(f"  - {performance_plot_path}")
    print(f"  - {report_path}")
    print(f"  - {data_filepath}")


if __name__ == "__main__":
    main()