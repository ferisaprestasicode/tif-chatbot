# main.py
import os
import logging
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to sys.path
current_dir = Path(__file__).resolve().parent
parent_dir = str(current_dir.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data_loader import RRDDataLoader  
from models import AnomalyDetector

class AnomalyDetectionPipeline:
    """Main pipeline for traffic anomaly detection."""
    
    def __init__(self, base_dir: str = 'model/anomalies/output'):
        """Initialize the pipeline with output directory configuration."""
        self.base_dir = base_dir
        self.detector = AnomalyDetector()
        self._setup_logging()
        self._create_directories()

    def _setup_logging(self):
        """Configure logging with both file and console output."""
        log_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'anomaly_detection_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _create_directories(self):
        """Create all necessary output directories."""
        directories = [
            'reports',
            'model_checkpoints',
            'model_visualizations',
            'logs'
        ]
        
        for directory in directories:
            os.makedirs(os.path.join(self.base_dir, directory), exist_ok=True)

    def load_data(self, file_path: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load and validate the input data."""
        try:
            self.logger.info(f"Loading data from: {file_path}")
            data_loader = RRDDataLoader(file_path)
            
            if not data_loader.links:
                self.logger.error("No data loaded from RRD file")
                return None
                
            self.logger.info(f"Successfully loaded {len(data_loader.links)} links")
            return data_loader.links
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return None

    def generate_report(self, detailed_results: Dict) -> Optional[str]:
        """Generate a comprehensive evaluation report."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(
                self.base_dir, 
                'reports', 
                f'model_evaluation_report_{timestamp}.txt'
            )
            
            report_sections = self._create_report_sections(detailed_results)
            
            with open(report_file, 'w') as f:
                f.write('\n\n'.join(report_sections))
            
            self.logger.info(f"Report generated: {report_file}")
            return report_file
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return None

    def _create_report_sections(self, detailed_results: Dict) -> list:
        """Create all sections for the evaluation report."""
        sections = []
        
        # Header
        sections.append(f"""# Traffic Anomaly Detection Model Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}""")
        
        # Overall Statistics
        sections.append(self._create_overall_statistics(detailed_results))
        
        # Location Analysis
        sections.append(self._create_location_analysis(detailed_results))
        
        # Detailed Results
        sections.append(self._create_detailed_results(detailed_results))
        
        return sections

    def _create_overall_statistics(self, detailed_results: Dict) -> str:
        """Create the overall statistics section of the report."""
        model_counts = self._count_models(detailed_results)
        total_links = len(detailed_results)
        
        stats = [
            "## Overall Statistics",
            f"Total Links Analyzed: {total_links}\n",
            "### Model Distribution"
        ]
        
        for model, count in model_counts.items():
            percentage = (count / total_links) * 100
            stats.append(f"- {model}: {count} links ({percentage:.1f}%)")
        
        return '\n'.join(stats)

    def _create_location_analysis(self, detailed_results: Dict) -> str:
        """Create the location-specific analysis section of the report."""
        location_results = self._group_by_location(detailed_results)
        
        sections = ["## Location-Specific Analysis"]
        
        for location, links in location_results.items():
            sections.append(f"\n### {location}")
            sections.append(f"Number of links: {len(links)}")
            
            avg_f1 = np.mean([link['f1_score'] for link in links])
            avg_anomaly = np.mean([link['anomaly_rate'] for link in links])
            
            sections.append(f"Average F1 Score: {avg_f1:.3f}")
            sections.append(f"Average Anomaly Rate: {avg_anomaly:.1f}%")
            
            model_dist = self._count_models({link['link']: link for link in links})
            sections.append("Model Distribution:")
            
            for model, count in model_dist.items():
                percentage = (count / len(links)) * 100
                sections.append(f"- {model}: {count} links ({percentage:.1f}%)")
        
        return '\n'.join(sections)

    def _create_detailed_results(self, detailed_results: Dict) -> str:
        """Create the detailed results section of the report."""
        sections = ["## Detailed Results by Link"]
        
        for link_name, results in detailed_results.items():
            sections.extend([
                f"\n{link_name}:",
                f"  Best Model: {results['best_model']}",
                f"  F1 Score: {results['f1_score']:.3f}",
                f"  Anomaly Rate: {results['anomaly_rate']*100:.1f}%"
            ])
        
        return '\n'.join(sections)

    @staticmethod
    def _count_models(results: Dict) -> Dict:
        """Count the occurrences of each model type."""
        model_counts = {}
        for result in results.values():
            model = result['best_model']
            model_counts[model] = model_counts.get(model, 0) + 1
        return model_counts

    @staticmethod
    def _group_by_location(results: Dict) -> Dict:
        """Group results by location identifier."""
        location_results = {}
        for link_name, result in results.items():
            location = link_name.split('_')[1]  # Assumes link_name format includes location
            if location not in location_results:
                location_results[location] = []
            location_results[location].append({'link': link_name, **result})
        return location_results

    def run(self, data_path: str) -> Optional[str]:
        """Execute the complete anomaly detection pipeline."""
        try:
            self.logger.info("Starting anomaly detection pipeline...")
            
            # Load data
            links_data = self.load_data(data_path)
            if not links_data:
                return None
            
            # Run analysis
            self.logger.info("Running anomaly detection models...")
            all_results = self.detector.evaluate_all_links(links_data)
            
            # Generate report
            self.logger.info("Generating evaluation report...")
            report_file = self.generate_report(self.detector.detailed_results)
            
            if report_file:
                self.logger.info("\nAnalysis complete. Output locations:")
                self.logger.info(f"- Report: {report_file}")
                self.logger.info(f"- Model checkpoints: {os.path.join(self.base_dir, 'model_checkpoints')}")
                self.logger.info(f"- Visualizations: {os.path.join(self.base_dir, 'model_visualizations')}")
                self.logger.info(f"- Logs: {os.path.join(self.base_dir, 'logs')}")
            
            return report_file
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return None

def main():
    """Entry point for the anomaly detection pipeline."""
    # Configuration
    data_path = "data/processed/rrd_log_minutes.txt"
    base_dir = "model/anomalies/output"
    
    # Create and run pipeline
    try:
        pipeline = AnomalyDetectionPipeline(base_dir=base_dir)
        report_file = pipeline.run(data_path)
        
        if report_file:
            print(f"\nPipeline completed successfully. Report saved to: {report_file}")
        else:
            print("\nPipeline failed to complete. Check the logs for details.")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
