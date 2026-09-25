import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from src.evaluate import evaluate_macro_f05
import gc
import logging

logger = logging.getLogger(__name__)

class EntityMatchingModel:
    def __init__(self):
        self.model = None
        self.best_threshold = 0.5
        self.features = None

    def prepare_training_data(self, df_features: pd.DataFrame, df_ground_truth: pd.DataFrame):
        """
        Merges generated features with ground truth to create binary labels.
        """
        logger.info("Preparing training data labels...")
        # Create a mapping from s1_id to a set of matched candidate ids
        gt_mapping = {}
        # Filter ground truth to only S1 IDs we care about for speed
        s1_ids_in_features = set(df_features['source1_entity_id'].unique())
        df_gt_filtered = df_ground_truth[df_ground_truth['source1_entity_id'].isin(s1_ids_in_features)]
        
        for s1_id, matched_str in zip(df_gt_filtered['source1_entity_id'], df_gt_filtered['matched_entity_ids']):
            if pd.isna(matched_str) or matched_str == "":
                gt_mapping[s1_id] = set()
            else:
                gt_mapping[s1_id] = set(str(matched_str).split(','))
                
        # Generate labels
        labels = []
        for s1_id, cand_id in zip(df_features['source1_entity_id'], df_features['candidate_entity_id']):
            is_match = 1 if (s1_id in gt_mapping and cand_id in gt_mapping[s1_id]) else 0
            labels.append(is_match)
            
        df_features['label'] = labels
        return df_features

    def train_cv(self, df: pd.DataFrame, n_splits=3):
        """
        Trains model using GroupKFold to prevent leakage across s1_ids.
        Tunes threshold to maximize Macro F0.5
        """
        feature_cols = [c for c in df.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label']]
        self.features = feature_cols
        
        logger.info(f"Training on {len(df)} pairs with {len(feature_cols)} features.")
        
        gkf = GroupKFold(n_splits=n_splits)
        groups = df['source1_entity_id'].values
        X = df[feature_cols].values
        y = df['label'].values
        
        oof_preds = np.zeros(len(df))
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            logger.info(f"Training Fold {fold+1}")
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'learning_rate': 0.1,
                'max_depth': 6,
                'num_leaves': 31,
                'verbose': -1,
                'n_jobs': -1,
                'seed': 42
            }
            
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[train_data, val_data],
                callbacks=callbacks
            )
            
            models.append(model)
            oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            
        self.models = models
        df['pred_prob'] = oof_preds
        
        # Optimize threshold
        self.best_threshold = self._optimize_threshold(df)
        logger.info(f"Best Threshold selected: {self.best_threshold}")
        return self.models

    def _optimize_threshold(self, df_oof: pd.DataFrame):
        thresholds = np.linspace(0.1, 0.9, 17)
        best_f05 = -1
        best_th = 0.5
        
        # We need the ground truth mapping
        # Let's aggregate ground truth from the dataframe
        df_gt = df_oof[df_oof['label'] == 1].groupby('source1_entity_id')['candidate_entity_id'].apply(list).reset_index()
        gt_dict = {row['source1_entity_id']: row['candidate_entity_id'] for _, row in df_gt.iterrows()}
        # For singletons in OOF, they might not have label=1
        all_s1 = df_oof['source1_entity_id'].unique()
        for s1 in all_s1:
            if s1 not in gt_dict:
                gt_dict[s1] = []
                
        for th in thresholds:
            df_pred = df_oof[df_oof['pred_prob'] >= th]
            pred_agg = df_pred.groupby('source1_entity_id')['candidate_entity_id'].apply(list).reset_index()
            pred_dict = {row['source1_entity_id']: row['candidate_entity_id'] for _, row in pred_agg.iterrows()}
            for s1 in all_s1:
                if s1 not in pred_dict:
                    pred_dict[s1] = []
                    
            f05 = evaluate_macro_f05(gt_dict, pred_dict)
            if f05 > best_f05:
                best_f05 = f05
                best_th = th
                
        return best_th

    def predict(self, df_features: pd.DataFrame):
        X = df_features[self.features].values
        preds = np.zeros(len(df_features))
        for model in self.models:
            preds += model.predict(X, num_iteration=model.best_iteration) / len(self.models)
        
        df_features['pred_prob'] = preds
        return df_features
