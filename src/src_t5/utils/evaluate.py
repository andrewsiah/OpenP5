import numpy as np
import math
import random
import time
import logging


def rel_results_filtered(user_positive, id2user, user_idx, return_num, predictions, targets, scores, k):
    """Calculate filtered relevance results."""
    start_time = time.time()
    
    results = []
    batch_length = len(targets)
    for b in range(batch_length):
        uidx = user_idx[b]
        user_id = id2user[uidx]
        positive = user_positive[user_id]
        one_batch_sequence = predictions[
            b * return_num : (b + 1) * return_num
        ]
        one_batch_score = scores[
            b * return_num : (b + 1) * return_num
        ]
        pairs = [(a, b) for a, b in zip(one_batch_sequence, one_batch_score)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        gt = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] not in positive:
                if sorted_pred[0] == gt:
                    one_results.append(1)
                else:
                    one_results.append(0)
                if len(one_results) >= k:
                    break
            else:
                continue
        
        results.append(one_results)

    process_time = time.time() - start_time
    if process_time > 0.1:  # Only log if processing takes significant time
        logging.debug(f"rel_results_filtered processing time: {process_time:.3f}s for batch_size={batch_length}")
    
    return results

def rel_results(predictions, targets, scores, k):
    """Calculate relevance results between generated and gold sentences."""
    start_time = time.time()
    
    results = []
    batch_length = len(targets)
    for b in range(batch_length):
        one_batch_sequence = predictions[
            b * k : (b + 1) * k
        ]
        one_batch_score = scores[
            b * k : (b + 1) * k
        ]
        pairs = [(a, b) for a, b in zip(one_batch_sequence, one_batch_score)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        gt = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == gt:
                one_results.append(1)
            else:
                one_results.append(0)
        
        results.append(one_results)
    
    process_time = time.time() - start_time
    if process_time > 0.1:  # Only log if processing takes significant time
        logging.debug(f"rel_results processing time: {process_time:.3f}s for batch_size={batch_length}")
    
    return results

def get_metrics_results(rel_results, metrics):
    """Calculate metric scores from relevance results."""
    start_time = time.time()
    
    res = []
    for m in metrics:
        metric_start = time.time()
        if m.lower().startswith('hit'):
            k = int(m.split('@')[1])
            res.append(hit_at_k(rel_results, k))
        elif m.lower().startswith('ndcg'):
            k = int(m.split('@')[1])
            res.append(ndcg_at_k(rel_results, k))
        metric_time = time.time() - metric_start
        if metric_time > 0.05:  # Log individual metric timing if significant
            logging.debug(f"Metric {m} calculation time: {metric_time:.3f}s")
    
    process_time = time.time() - start_time
    if process_time > 0.1:  # Only log if total processing takes significant time
        logging.debug(f"get_metrics_results total processing time: {process_time:.3f}s")
    
    return np.array(res)

def ndcg_at_k(relevance, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    for row in relevance:
        rel = row[:k]
        one_ndcg = 0.0
        for i in range(len(rel)):
            one_ndcg += rel[i] / math.log(i+2,2)
        ndcg += one_ndcg
    return ndcg
        
    
def hit_at_k(relevance, k):
    correct = 0.0
    for row in relevance:
        rel = row[:k]
        if sum(rel) > 0:
            correct += 1
    return correct
        