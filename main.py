import os
import sys
import logging
import subprocess
from pathlib import Path
import pandas as pd

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.blocking import Blocker
from src.feature_extraction import FeatureExtractor
from src.model import EntityMatchingModel
from src.postprocessor import Postprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_pipeline(split, loader, preprocessor, blocker, feat_extractor, model, is_train=True, gt_path=None):
    logger.info(f"--- STARTING {split.upper()} PHASE ---")
    df_s1, df_s2, df_s3 = loader.load_all_sources(split, use_polars=False)
    
    df_s1 = preprocessor.preprocess_df_pandas(df_s1)
    df_s2 = preprocessor.preprocess_df_pandas(df_s2)
    df_s3 = preprocessor.preprocess_df_pandas(df_s3)
    

    
    logger.info("Generating candidate pairs...")
    df_cands = blocker.generate_candidate_pairs(df_s1, df_s2, df_s3)
    
    logger.info("Extracting features...")
    df_candidates_all = pd.concat([df_s2, df_s3], ignore_index=True)
    df_features = feat_extractor.extract_features(df_cands, df_s1, df_candidates_all)
    
    if is_train:
        df_gt = pd.read_csv(gt_path, sep='\t', dtype=str, quoting=3, on_bad_lines="skip")
        df_features = model.prepare_training_data(df_features, df_gt)
        logger.info("Training model...")
        model.train_cv(df_features, n_splits=3)
        return None, None
    else:
        logger.info("Predicting...")
        df_preds = model.predict(df_features)
        return df_cands, df_preds, df_s1

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument('--dataset_dir', type=str, default="/Users/devpopli/AMAZON ML/student_resource/dataset", help="Path to the dataset directory")
    parser.add_argument('--validate_script', type=str, default="/Users/devpopli/AMAZON ML/student_resource/utils/validate_submission.py", help="Path to validate_submission.py")
    parser.add_argument('--project_root', type=str, default="/Users/devpopli/AMAZON ML/code/business_entity_resolution", help="Path to project root directory")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    dataset_dir = args.dataset_dir
    output_dir = project_root / "output"
    
    loader = DataLoader(dataset_dir)
    preprocessor = Preprocessor()
    blocker = Blocker(top_k=15)  # bumped from 5 -- was too tight once blocking stabilized
    feat_extractor = FeatureExtractor()
    model = EntityMatchingModel()
    
    # Create output dir if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    postprocessor = Postprocessor(str(output_dir))
    
    gt_path = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")
    
    
    # Train
    process_pipeline("train", loader, preprocessor, blocker, feat_extractor, model, is_train=True, gt_path=gt_path)
    
    # Test - We must process all S1 test data to pass validation. 
    # Warning: this might take a lot of RAM/Time. 
    df_cands, df_preds, df_s1 = process_pipeline("test", loader, preprocessor, blocker, feat_extractor, model, is_train=False)
    
    # Output
    logger.info("--- STARTING OUTPUT PHASE ---")
    postprocessor.write_candidates(df_cands, df_s1)
    postprocessor.write_matching_results(df_preds, df_s1, model.best_threshold)
    
    # Validate
    logger.info("--- RUNNING VALIDATION SCRIPT ---")
    validate_script = args.validate_script
    cmd = [
        "python3", validate_script,
        "--matching", str(output_dir / "matching_results.tsv"),
        "--candidate", str(output_dir / "candidate_pairs.tsv"),
        "--test-dir", os.path.join(dataset_dir, "test")
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    logger.info("Validation STDOUT:\n" + result.stdout)
    if result.stderr:
        logger.error("Validation STDERR:\n" + result.stderr)
        
    if result.returncode == 0:
        logger.info("Pipeline executed and validated successfully!")
    else:
        logger.error("Validation failed.")

if __name__ == "__main__":
    main()
