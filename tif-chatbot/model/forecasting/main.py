# File: main.py
from collections import defaultdict
from datetime import datetime  
import numpy as np 
from data_loader import RRDDataLoader
from models import ModelComparison
from utils import ModelReporter
import os
import pandas as pd

def create_directories():
    """Create necessary directories for outputs"""
    base_dir = 'model/forecasting/output'
    directories = [
        'reports',
        'model_checkpoints',
        'model_visualizations'
    ]
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create other directories inside base directory
    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)

def generate_report(detailed_results, output_file="model_evaluation_report.txt"):
    try:
        # Ensure output directory exists and create full path
        base_dir = 'model/forecasting/output/reports'
        os.makedirs(base_dir, exist_ok=True)
        output_path = os.path.join(base_dir, output_file)
        
        # Group links by location
        location_results = defaultdict(list)
        for link_name, results in detailed_results.items():
            location = link_name.split('_')[1]
            location_results[location].append({
                'link': link_name,
                **results
            })
        
        # Calculate overall statistics
        model_counts = defaultdict(int)
        for results in detailed_results.values():
            model_counts[results['best_model']] += 1
        
        # Generate report content
        report_content = f"""# Traffic Anomaly Detection Model Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Statistics
Total Links Analyzed: {len(detailed_results)}

### Model Distribution
"""
        
        # Add model distribution
        total_links = len(detailed_results)
        for model, count in model_counts.items():
            percentage = (count / total_links) * 100
            report_content += f"- {model}: {count} links ({percentage:.1f}%)\n"
        
        # Add location-specific analysis
        report_content += "\n## Location-Specific Analysis\n"
        for location, links in location_results.items():
            report_content += f"\n### {location}\n"
            report_content += f"Number of links: {len(links)}\n"
            
            # Calculate location statistics
            location_models = defaultdict(int)
            avg_f1 = np.mean([link.get('f1_score', 0) for link in links])
            avg_anomaly_rate = np.mean([link.get('anomaly_rate', 0) for link in links])
            
            for link in links:
                location_models[link['best_model']] += 1
            
            report_content += f"Average F1 Score: {avg_f1:.3f}\n"
            report_content += f"Average Anomaly Rate: {avg_anomaly_rate:.1f}%\n"
            report_content += "Model Distribution:\n"
            
            for model, count in location_models.items():
                percentage = (count / len(links)) * 100
                report_content += f"- {model}: {count} links ({percentage:.1f}%)\n"
        
        # Add detailed results for each link
        report_content += "\n## Detailed Results by Link\n"
        for link_name, results in detailed_results.items():
            report_content += f"\n{link_name}:\n"
            report_content += f"  Best Model: {results['best_model']}\n"
            report_content += f"  F1 Score: {results.get('f1_score', 0):.3f}\n"
            report_content += f"  Anomaly Rate: {results.get('anomaly_rate', 0)*100:.1f}%\n"
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report_content)
            
        print(f"\nReport saved to {output_path}")
        return report_content
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None
    
def main():
    try:
        print("Starting traffic analysis...")
        
        # Create necessary directories
        create_directories()
        print("Directories created")
        
        # Load RRD data
        file_path = "data/processed/rrd_log.txt"  
        print(f"Loading data from {file_path}")
        data_loader = RRDDataLoader(file_path)
        
        if not data_loader.links:
            print("No data loaded from RRD file")
            return
        print(f"Loaded {len(data_loader.links)} links")
        
        # Initialize and run comparison
        comparison = ModelComparison()
        reporter = ModelReporter()
        print("Running model comparison...")
        all_results = comparison.evaluate_all_links(data_loader.links)
        print(f"Comparison complete. Results: {len(all_results)} links")
        print(f"Detailed results: {len(comparison.detailed_results)} items")
        
        # Generate report
        report_file = "model_evaluation_report.txt"
        print(f"Generating report to {report_file}")
        report_content = generate_report(comparison.detailed_results, report_file)
        print(f"Report generation {'succeeded' if report_content else 'failed'}")
        
        print("\nAnalysis complete")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()

## add minio config 


# from collections import defaultdict
# from datetime import datetime  
# import numpy as np 
# from data_loader import RRDDataLoader
# from models import ModelComparison
# from utils import ModelReporter
# import os
# import pandas as pd
# from io import BytesIO
# import json
# from minio import Minio

# # MinIO Configuration
# MINIO_ENDPOINT = "10.62.179.53:9000"
# MINIO_ACCESS_KEY = "pgF2QVgkdP5TucUv34Zm"
# MINIO_SECRET_KEY = "tzCS5tVg6W8XpAXYSMoXMDRdo0W83IpTOhvFE0LH"
# MINIO_SECURE = False
# MINIO_BUCKET = "traffic-data"

# class MinioHandler:
#     def __init__(self):
#         self.client = Minio(
#             MINIO_ENDPOINT,
#             access_key=MINIO_ACCESS_KEY,
#             secret_key=MINIO_SECRET_KEY,
#             secure=MINIO_SECURE
#         )
#         self.ensure_bucket_exists()
    
#     def ensure_bucket_exists(self):
#         """Ensure the bucket exists, create if it doesn't"""
#         if not self.client.bucket_exists(MINIO_BUCKET):
#             self.client.make_bucket(MINIO_BUCKET)
    
#     def create_directories(self, base_path, directories):
#         """Create directory structure in MinIO (not actually needed but kept for compatibility)"""
#         # MinIO doesn't need directory creation, but we'll keep track of paths
#         self.paths = [f"{base_path}/{directory}/" for directory in directories]
    
#     def save_file(self, content, path):
#         """Save content to MinIO"""
#         try:
#             data_bytes = content.encode('utf-8')
#             data_stream = BytesIO(data_bytes)
#             self.client.put_object(
#                 MINIO_BUCKET,
#                 path,
#                 data_stream,
#                 len(data_bytes),
#                 content_type='text/plain'
#             )
#             return True
#         except Exception as e:
#             print(f"Error saving to MinIO: {e}")
#             return False
    
#     def read_file(self, path):
#         """Read content from MinIO"""
#         try:
#             data = self.client.get_object(MINIO_BUCKET, path)
#             return data.read().decode('utf-8')
#         except Exception as e:
#             print(f"Error reading from MinIO: {e}")
#             return None

# def create_directories():
#     """Create necessary directories in MinIO"""
#     minio_handler = MinioHandler()
#     base_dir = 'model/forecasting/output'
#     directories = [
#         'reports',
#         'model_checkpoints',
#         'model_visualizations'
#     ]
#     minio_handler.create_directories(base_dir, directories)

# def generate_report(detailed_results, output_file="model_evaluation_report.txt"):
#     try:
#         minio_handler = MinioHandler()
#         base_dir = 'model/forecasting/output/reports'
#         output_path = f"{base_dir}/{output_file}"
        
#         # Group links by location
#         location_results = defaultdict(list)
#         for link_name, results in detailed_results.items():
#             location = link_name.split('_')[1]
#             location_results[location].append({
#                 'link': link_name,
#                 **results
#             })
        
#         # Calculate overall statistics
#         model_counts = defaultdict(int)
#         for results in detailed_results.values():
#             model_counts[results['best_model']] += 1
        
#         # Generate report content
#         report_content = f"""# Traffic Anomaly Detection Model Evaluation Report
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# ## Overall Statistics
# Total Links Analyzed: {len(detailed_results)}

# ### Model Distribution
# """
        
#         # Add model distribution
#         total_links = len(detailed_results)
#         for model, count in model_counts.items():
#             percentage = (count / total_links) * 100
#             report_content += f"- {model}: {count} links ({percentage:.1f}%)\n"
        
#         # Add location-specific analysis
#         report_content += "\n## Location-Specific Analysis\n"
#         for location, links in location_results.items():
#             report_content += f"\n### {location}\n"
#             report_content += f"Number of links: {len(links)}\n"
            
#             # Calculate location statistics
#             location_models = defaultdict(int)
#             avg_f1 = np.mean([link.get('f1_score', 0) for link in links])
#             avg_anomaly_rate = np.mean([link.get('anomaly_rate', 0) for link in links])
            
#             for link in links:
#                 location_models[link['best_model']] += 1
            
#             report_content += f"Average F1 Score: {avg_f1:.3f}\n"
#             report_content += f"Average Anomaly Rate: {avg_anomaly_rate:.1f}%\n"
#             report_content += "Model Distribution:\n"
            
#             for model, count in location_models.items():
#                 percentage = (count / len(links)) * 100
#                 report_content += f"- {model}: {count} links ({percentage:.1f}%)\n"
        
#         # Add detailed results for each link
#         report_content += "\n## Detailed Results by Link\n"
#         for link_name, results in detailed_results.items():
#             report_content += f"\n{link_name}:\n"
#             report_content += f"  Best Model: {results['best_model']}\n"
#             report_content += f"  F1 Score: {results.get('f1_score', 0):.3f}\n"
#             report_content += f"  Anomaly Rate: {results.get('anomaly_rate', 0)*100:.1f}%\n"
        
#         # Save report to MinIO
#         if minio_handler.save_file(report_content, output_path):
#             print(f"\nReport saved to MinIO: {output_path}")
#             return report_content
#         return None
            
#     except Exception as e:
#         print(f"Error generating report: {str(e)}")
#         return None
    
# def main():
#     try:
#         print("Starting traffic analysis...")
        
#         # Create necessary directories in MinIO
#         create_directories()
#         print("MinIO directory structure initialized")
        
#         # Load RRD data from MinIO
#         file_path = "data/processed/rrd_log.txt"
#         print(f"Loading data from MinIO: {file_path}")
#         minio_handler = MinioHandler()
#         rrd_data = minio_handler.read_file(file_path)
        
#         if not rrd_data:
#             print("No data loaded from MinIO")
#             return
            
#         data_loader = RRDDataLoader(rrd_data, from_string=True)  # Assuming RRDDataLoader is modified to accept string data
        
#         if not data_loader.links:
#             print("No links found in data")
#             return
#         print(f"Loaded {len(data_loader.links)} links")
        
#         # Initialize and run comparison
#         comparison = ModelComparison()
#         reporter = ModelReporter()
#         print("Running model comparison...")
#         all_results = comparison.evaluate_all_links(data_loader.links)
#         print(f"Comparison complete. Results: {len(all_results)} links")
#         print(f"Detailed results: {len(comparison.detailed_results)} items")
        
#         # Generate report
#         report_file = "model_evaluation_report.txt"
#         print(f"Generating report to MinIO: {report_file}")
#         report_content = generate_report(comparison.detailed_results, report_file)
#         print(f"Report generation {'succeeded' if report_content else 'failed'}")
        
#         print("\nAnalysis complete")
        
#     except Exception as e:
#         print(f"Error in main execution: {str(e)}")
#         import traceback
#         print(traceback.format_exc())

# if __name__ == "__main__":
#     main()
