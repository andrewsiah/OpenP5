Goal: I want to test how data mixture affects the performance of the model on all the downstream tasks in the evals.

I need a mixture of data and evals.

e.g. for Beauty, CDs, Clothing, Electronics and Movies dataset. 
For the current setup of running on sequential data.


1. I want to run eval on all 5 data mixtures at every epoch.
2. I want to pre-make the data mixtures for all 5 datasets. The mixture is a constant amount of data that have different weights for each dataset. The weights are drawn from a dirichlet distribution. e.g. if i want to run 100 experiments, i want to make 100 mixtures of data for all 5 datasets.

Let's break this down into manageable steps:

### Phase 1: Multi-Dataset Evaluation
1. **Modify SingleRunner to support multi-dataset evaluation**
   - Add a method to load test loaders for multiple datasets
   - Create a method to run evaluation on all datasets
   - Modify logging to track per-dataset metrics

2. **Update argument parsing**
   - Add arguments for eval datasets
   - Add arguments for eval paths
   - Add arguments for logging multi-dataset results

3. **Implement logging structure**
   - Create a consistent format for multi-dataset results
   - Set up proper file paths for each dataset's results

### Phase 2: Data Mixture Training
1. **Create Mixture Generator**
   - Implement Dirichlet sampling
   - Create data sampling based on weights
   - Save mixture configurations for reproducibility

2. **Data Loading Pipeline**
   - Create mixture dataset class
   - Implement weighted sampling
   - Handle different dataset sizes

3. **Training Loop**
   - Modify training to use mixture data
   - Track per-dataset performance during training
   - Save mixture-specific checkpoints

Let's start with Phase 1. Here's the detailed implementation plan for Step 1:

1. **Modify SingleRunner**:
```python
class SingleRunner:
    def __init__(self, model, tokenizer, train_loader, valid_loader, device, args):
        # Existing init code...
        self.eval_datasets = args.eval_datasets.split(',')
        self.test_loaders = self.get_all_test_loaders()
    
    def get_all_test_loaders(self):
        """Load test loaders for all evaluation datasets"""
        test_loaders = {}
        for dataset in self.eval_datasets:
            test_loaders[dataset] = self.get_test_loader(dataset)
        return test_loaders
    
    def evaluate_all_datasets(self):
        """Run evaluation on all datasets"""
        results = {}
        for dataset, loader in self.test_loaders.items():
            logging.info(f"Evaluating on {dataset}")
            results[dataset] = self.test_dataset_task(loader)
        return results
```

Would you like me to continue with the implementation details for any of these steps?
