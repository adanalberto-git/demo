import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def load_and_analyze_gaze_data(file_path):
    try:
        print(f"Attempting to load data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows of data")
        
        # Calculate velocities
        df['dx'] = df['gx'].diff()
        df['dy'] = df['gy'].diff()
        df['dt'] = df['t'].diff()
        df['velocity'] = np.sqrt(df['dx']**2 + df['dy']**2) / df['dt']
        
        # Prepare features for ML
        features = df[['gx', 'gy', 'velocity']].fillna(0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # 1. DBSCAN for fixation clustering
        dbscan = DBSCAN(eps=0.01, min_samples=3)
        df['fixation_clusters'] = dbscan.fit_predict(scaled_features[:, :2])
        
        # 2. K-means for identifying areas of interest
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['attention_areas'] = kmeans.fit_predict(scaled_features[:, :2])
        
        # 3. Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        df['anomalies'] = iso_forest.fit_predict(scaled_features)
        
        # 4. PCA for dimension reduction
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        df['pca1'] = pca_result[:, 0]
        df['pca2'] = pca_result[:, 1]
        
        # 5. SVM for movement classification
        y = (features['velocity'] > features['velocity'].mean()).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, y, test_size=0.2, random_state=42
        )
        
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train, y_train)
        df['svm_movement_class'] = svm.predict(scaled_features)
        
        return df, pca.explained_variance_ratio_
    
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

def print_analysis_summary(df, pca_variance):
    if df is not None and pca_variance is not None:
        print("\nAnalysis Results:")
        print(f"Number of samples analyzed: {len(df)}")
        print(f"Recording duration: {df['t'].max() - df['t'].min():.2f} seconds")
        print(f"Number of fixation clusters: {len(df['fixation_clusters'].unique())}")
        print(f"Percentage of anomalies: {(df['anomalies'] == -1).mean()*100:.1f}%")
        print(f"PCA explained variance: {pca_variance[0]*100:.1f}% and {pca_variance[1]*100:.1f}%")
        print("\nMovement types distribution:")
        print(df['svm_movement_class'].value_counts(normalize=True).mul(100).round(1))


def plot_gaze_analysis(df, save_path='gaze_analysis_plots.png'):
    """
    Create and save visualizations of the gaze analysis results
    
    Parameters:
    df : pandas DataFrame
        The analyzed gaze data
    save_path : str
        Path where to save the plot (default: 'gaze_analysis_plots.png')
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Fixation Clusters
        axes[0,0].scatter(df['gx'].values, df['gy'].values, 
                         c=df['fixation_clusters'].values, 
                         cmap='viridis',
                         alpha=0.6)
        axes[0,0].set_title('Fixation Clusters')
        axes[0,0].set_xlabel('Horizontal Gaze Position')
        axes[0,0].set_ylabel('Vertical Gaze Position')
        
        # Plot 2: Attention Areas
        axes[0,1].scatter(df['gx'].values, df['gy'].values, 
                         c=df['attention_areas'].values, 
                         cmap='plasma',
                         alpha=0.6)
        axes[0,1].set_title('Attention Areas')
        axes[0,1].set_xlabel('Horizontal Gaze Position')
        axes[0,1].set_ylabel('Vertical Gaze Position')
        
        # Plot 3: Velocity Profile
        axes[0,2].plot(df['t'].values, df['velocity'].values, 'b-', alpha=0.5)
        axes[0,2].set_title('Velocity Profile')
        axes[0,2].set_xlabel('Time (s)')
        axes[0,2].set_ylabel('Velocity')
        
        # Plot 4: Anomaly Detection
        axes[1,0].scatter(df['gx'].values, df['gy'].values, 
                         c=df['anomalies'].values, 
                         cmap='RdYlBu',
                         alpha=0.6)
        axes[1,0].set_title('Detected Anomalies')
        axes[1,0].set_xlabel('Horizontal Gaze Position')
        axes[1,0].set_ylabel('Vertical Gaze Position')
        
        # Plot 5: PCA Results
        axes[1,1].scatter(df['pca1'].values, df['pca2'].values, 
                         c=df['velocity'].values, 
                         cmap='viridis',
                         alpha=0.6)
        axes[1,1].set_title('PCA Pattern Analysis')
        axes[1,1].set_xlabel('First Principal Component')
        axes[1,1].set_ylabel('Second Principal Component')
        
        # Plot 6: SVM Classification
        axes[1,2].scatter(df['gx'].values, df['gy'].values, 
                         c=df['svm_movement_class'].values, 
                         cmap='RdYlBu',
                         alpha=0.6)
        axes[1,2].set_title('Movement Classification')
        axes[1,2].set_xlabel('Horizontal Gaze Position')
        axes[1,2].set_ylabel('Vertical Gaze Position')
        
        plt.tight_layout()
        
        # Save the figure
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Display the plot
        plt.show()
        
        print(f"Plot successfully saved to {save_path}")
        return fig
    
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        return None

# In the main execution part, modify it to include saving:
if __name__ == "__main__":
    file_path = "./RBU_FLAT1_GAZE.csv"
    
    # Create output directory if it doesn't exist
    import os
    output_dir = "gaze_analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Analyze data
    analyzed_df, pca_variance = load_and_analyze_gaze_data(file_path)
    
    if analyzed_df is not None:
        # Save plots with different formats and in the output directory
        plot_gaze_analysis(analyzed_df, os.path.join(output_dir, 'gaze_analysis_plots.png'))  # PNG format
        plot_gaze_analysis(analyzed_df, os.path.join(output_dir, 'gaze_analysis_plots.pdf'))  # PDF format
        print_analysis_summary(analyzed_df, pca_variance)