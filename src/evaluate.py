def evaluate_macro_f05(gt_dict: dict, pred_dict: dict):
    """
    gt_dict: dict mapping s1_id to list/set of true matching cand_ids
    pred_dict: dict mapping s1_id to list/set of predicted cand_ids
    Singletons: If true is empty and pred is empty -> score 1.0
                If true is empty and pred is not empty -> score 0.0
    """
    f05_scores = []
    
    for s1_id, gt_cands in gt_dict.items():
        pred_cands = pred_dict.get(s1_id, [])
        
        gt_set = set(gt_cands)
        pred_set = set(pred_cands)
        
        if len(gt_set) == 0:
            if len(pred_set) == 0:
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
            continue
            
        if len(pred_set) == 0:
            f05_scores.append(0.0)
            continue
            
        tp = len(gt_set.intersection(pred_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            f05_scores.append(0.0)
        else:
            # F_{0.5} = (1.25 * P * R) / (0.25 * P + R)
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            f05_scores.append(f05)
            
    if not f05_scores:
        return 0.0
        
    return sum(f05_scores) / len(f05_scores)
