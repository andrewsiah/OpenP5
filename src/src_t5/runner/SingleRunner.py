import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
from tqdm import tqdm
from utils import utils
import utils.generation_trie as gt
from data.TestDataset import TestDataset
from torch.utils.data import DataLoader
from processor.Collator import Collator, TestCollator
import time
import pdb
import numpy as np
import utils.evaluate as evaluate
import torch.distributed as dist
from collections import defaultdict
import json
import wandb
import os

class SingleRunner:
    def parse_runner_args(parser):
        """
        Parse dataset related command line arguments.
        
        Args:
            parser: ArgumentParser object to add arguments to
            
        Returns:
            parser: ArgumentParser with added arguments
        """
        parser.add_argument("--optim", type=str, default='AdamW', help='The name of the optimizer')
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--clip", type=float, default=1)
        parser.add_argument("--logging_step", type=int, default=100)
        parser.add_argument("--warmup_prop", type=float, default=0.05)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--adam_eps", type=float, default=1e-6)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--alpha", type=float, default=2)
        parser.add_argument("--train", type=int, default=1, help='train or not')
        parser.add_argument("--backbone", type=str, default='t5-small', help='backbone model name')
        parser.add_argument("--metrics", type=str, default='hit@5,hit@10,ndcg@5,ndcg@10', help='Metrics used for evaluation')
        parser.add_argument("--load", type=int, default=0, help='load model from model path or not.')
        parser.add_argument("--random_initialize", type=int, default=1, help='Randomly initialize number-related tokens.')
        parser.add_argument("--test_epoch", type=int, default=1, help='test once for how many epochs, 0 for no test during training.')
        parser.add_argument("--valid_select", type=int, default=0, help='use validation loss to select models')
        parser.add_argument("--test_before_train", type=int, default=1, help='whether test before training')
        parser.add_argument("--test_filtered", type=int, default=0, help='whether filter out the items in the training data.')
        parser.add_argument("--test_filtered_batch", type=int, default=1, help='whether testing with filtered data in batch.')
        parser.add_argument(
            "--eval_datasets",
            type=str,
            default=None,
            help="Comma-separated list of datasets to evaluate on. If not specified, uses training datasets"
        )
        parser.add_argument(
            "--use_wandb",
            type=int,
            default=0,
            help="Whether to use Weights & Biases logging"
        )
        parser.add_argument(
            "--wandb_project",
            type=str,
            default="OpenP5",
            help="Weights & Biases project name"
        )
        parser.add_argument(
            "--wandb_name",
            type=str,
            default=None,
            help="Weights & Biases run name. If None, will use model path basename"
        )
        
        return parser
    
    def __init__(self, model, tokenizer, train_loader, valid_loader, device, args):
        """
        Initialize the SingleRunner.
        
        Args:
            model: The model to train/evaluate
            tokenizer: Tokenizer for processing text
            train_loader: DataLoader for training data
            valid_loader: DataLoader for validation data  
            device: Device to run on (cuda/cpu)
            args: Arguments containing training parameters
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.args = args
        self.regenerate_candidate = 'candidate_items' in self.train_loader.dataset.datasets[0].info
        self.reconstruct_data = self.args.sample_prompt
        self.test_epoch = self.args.test_epoch
        self.valid_select = self.args.valid_select
        self.test_before_train = self.args.test_before_train
        self.test_filtered = self.args.test_filtered
        self.test_filtered_batch = self.args.test_filtered_batch
        
        self.get_testloader()
        
        if args.train:
            self.optimizer, self.scheduler = self.create_optimizer_and_scheduler()
            
        self.metrics = args.metrics.split(',')
        self.generate_num = max([int(m.split('@')[1]) for m in self.metrics])
        
        
    def train(self):
        """
        Train the model for the specified number of epochs.
        
        Handles training loop, validation, testing, and model checkpointing.
        """
        self.model.zero_grad()
        train_losses = []
        valid_losses = []
        best_epoch = -1
        
        if self.test_before_train > 0:
            self.test()
        for epoch in range(self.args.epochs):
            if self.regenerate_candidate:
                for ds in self.train_loader.dataset.datasets:
                    ds.generate_candidates()
                    ds.construct_sentence()
            elif self.reconstruct_data:
                for ds in self.train_loader.dataset.datasets:
                    ds.construct_sentence()
            self.train_loader.sampler.set_epoch(epoch)
            logging.info(f"Start training for epoch {epoch+1}")
            self.model.train()
            losses = []
            for batch in tqdm(self.train_loader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                
                output = self.model(
                    input_ids=input_ids,
                    whole_word_ids=whole_input_ids,
                    attention_mask=attn,
                    labels=output_ids,
                    alpha=self.args.alpha,
                    return_dict=True,
                )
                # compute loss masking padded tokens
                loss = output["loss"]
                lm_mask = output_attention != 0
                lm_mask = lm_mask.float()
                B, L = output_ids.size()
                loss = loss.view(B, L) * lm_mask
                loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

                # update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                losses.append(loss)
            train_epoch_loss = sum(losses)/len(losses)
            train_losses.append(train_epoch_loss)
            logging.info(f"The average training loss for epoch {epoch+1} is {train_epoch_loss}")
            
            self.test()
            
            if self.valid_select > 0:
                logging.info(f"Start validation for epoch {epoch+1}")
                losses = []
                self.model.eval()
                with torch.no_grad():
                    for batch in tqdm(self.valid_loader):
                        input_ids = batch[0].to(self.device)
                        attn = batch[1].to(self.device)
                        whole_input_ids = batch[2].to(self.device)
                        output_ids = batch[3].to(self.device)
                        output_attention = batch[4].to(self.device)

                        output = self.model(
                            input_ids=input_ids,
                            whole_word_ids=whole_input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=self.args.alpha,
                            return_dict=True,
                        )
                        # compute loss masking padded tokens
                        loss = output["loss"]
                        lm_mask = output_attention != 0
                        lm_mask = lm_mask.float()
                        B, L = output_ids.size()
                        loss = loss.view(B, L) * lm_mask
                        loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

                        losses.append(loss)
                    valid_epoch_loss = sum(losses)/len(losses)
                    valid_losses.append(valid_epoch_loss)
                    logging.info(f"The average valid loss for epoch {epoch+1} is {valid_epoch_loss}")

                    if valid_epoch_loss == min(valid_losses):
                        logging.info(f"The minimal validation loss so far.")
                        best_epoch = epoch + 1
                        utils.save_model(self.model, self.args.model_path)
                        logging.info(f"Save the current model to {self.args.model_path}")
                
            if self.test_epoch > 0:
                if (epoch + 1) % self.test_epoch == 0:
                    self.model.eval()
                    self.test()
            
        if self.valid_select > 0:
            if self.rank == 0:
                logging.info(f"The best validation at Epoch {best_epoch}")
        else:
            if self.rank == 0:
                torch.save(self.model.state_dict(), self.args.model_path)
                logging.info(f"Save the current model to {self.args.model_path}")
                
        return
    
    def create_optimizer_and_scheduler(self):
        """
        Create optimizer and learning rate scheduler.
        
        Returns:
            tuple: (optimizer, scheduler)
        """
        if self.args.rank == 0:
            logging.info("Building Optimizer and Scheduler")
        batch_per_epoch = len(self.train_loader)
        total_steps = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
        warmup_steps = int(total_steps * self.args.warmup_prop)
        
        if self.args.rank == 0:
            logging.info(f'Batch per epoch: {batch_per_epoch}')
            logging.info(f'Total steps: {total_steps}')
            logging.info(f'Warmup proportion: {self.args.warmup_prop}')
            logging.info(f'Warm up steps: {warmup_steps}')

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.args.rank == 0:
            logging.info(f"Building Optimizer {self.args.optim}")
        
        if self.args.optim.lower() == 'adamw':
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_eps)
        else:
            raise NotImplementedError
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return optimizer, scheduler
    
    def get_testloader(self):
        """
        Create test data loaders for each dataset and task combination.
        Supports loading multiple evaluation datasets.
        """
        self.testloaders = []
        
        # Get all datasets to evaluate on
        if hasattr(self.args, 'eval_datasets'):
            eval_datasets = self.args.eval_datasets.split(',')
        else:
            # Fallback to training datasets if eval_datasets not specified
            eval_datasets = self.args.datasets.split(',')
        
        # Get all tasks to evaluate
        tasks = self.args.tasks.split(',')
        
        if self.test_filtered > 0:
            collator = TestCollator(self.tokenizer)
        else:
            collator = Collator(self.tokenizer)
        
        logging.info(f"Setting up evaluation for datasets: {eval_datasets}")
        logging.info(f"Tasks to evaluate: {tasks}")
        
        for dataset in eval_datasets:
            for task in tasks:
                logging.info(f"Creating test loader for {dataset} - {task}")
                testdata = TestDataset(self.args, dataset, task)
                testloader = DataLoader(
                    dataset=testdata,
                    batch_size=self.args.eval_batch_size,
                    collate_fn=collator,
                    shuffle=False
                )
                self.testloaders.append(testloader)
    
    def test(self, path=None):
        """
        Run evaluation on test datasets.
        
        Args:
            path: Optional path to load model weights from
        """
        print("Starting evaluation...") 
        logging.info("\nSTARTING EVALUATION")
        self.model.eval()
        if path:
            self.model.load_state_dict(torch.load(path, map_location=self.device))

        # Create dict to store results for each dataset/task
        all_results = defaultdict(dict)
        
        # Initialize wandb if requested
        if self.args.use_wandb:
            if not wandb.run:
                name = self.args.wandb_name or os.path.basename(self.args.model_path)
                wandb.init(
                    project=self.args.wandb_project,
                    name=name,
                    config=vars(self.args)
                )
        
        for loader in self.testloaders:
            dataset_name = loader.dataset.dataset
            task_name = loader.dataset.task
            
            if self.test_filtered > 0:
                if self.test_filtered_batch > 0:
                    metrics = self.test_dataset_task_filtered_batch(loader, return_metrics=True)
                else:
                    assert self.args.eval_batch_size == 1
                    metrics = self.test_dataset_task_filtered(loader, return_metrics=True)
            else:
                metrics = self.test_dataset_task(loader, return_metrics=True)
                
            # Store results
            metric_dict = {
                metric: float(value) for metric, value in zip(self.metrics, metrics)
            }
            all_results[dataset_name][task_name] = metric_dict
            
            # Log to wandb
            if self.args.use_wandb:
                wandb_metrics = {
                    f"{dataset_name}/{task_name}/{metric}": value 
                    for metric, value in metric_dict.items()
                }
                wandb.log(wandb_metrics)

        # Log all results
        logging.info("\nEVALUATION RESULTS:")
        logging.info(json.dumps(all_results, indent=2))
        
        # Save results to file
        results_path = self.args.model_path.replace('.pt', '_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"\nSaved detailed results to {results_path}")
        
        # Log summary metrics to wandb
        if self.args.use_wandb:
            # Calculate average metrics across datasets
            avg_metrics = defaultdict(list)
            for dataset_results in all_results.values():
                for task_results in dataset_results.values():
                    for metric, value in task_results.items():
                        avg_metrics[metric].append(value)
            
            wandb.log({
                f"avg_{metric}": np.mean(values)
                for metric, values in avg_metrics.items()
            })
        
        logging.info("EVALUATION COMPLETE\n")
        return all_results
    
    def test_dataset_task_filtered_batch(self, testloader, return_metrics=False):
        """
        Run filtered batch evaluation on a test dataset. This method evaluates the model's ability to predict items 
        while filtering out items that appear in the user's training history.
        
        The filtering is done in batches for efficiency. For each user batch, it:
        1. Gets the set of all possible items minus those in the user's history
        2. Generates predictions only from the filtered candidate set
        3. Evaluates metrics like Hit Rate and NDCG on the filtered predictions
        
        Args:
            testloader: DataLoader containing test data with user histories and ground truth items
        """
        dataset_name = testloader.dataset.dataset
        task_name = testloader.dataset.task
        logging.info(f'testing filtered {dataset_name} dataset on {task_name} task')
        
        # Start timing
        start_time = time.time()
        
        test_total = 0
        with torch.no_grad():
            candidates = testloader.dataset.all_items
            candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in candidates
                ]
            )
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
            
            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                user_idx = batch[5].to(self.device)
                
                prediction = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=30,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num + testloader.dataset.max_positive,
                        num_return_sequences=self.generate_num + testloader.dataset.max_positive,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]
                
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                
                rel_results = evaluate.rel_results_filtered(testloader.dataset.positive, testloader.dataset.id2user, user_idx, \
                                                            self.generate_num+testloader.dataset.max_positive, \
                                                            generated_sents, gold_sents, prediction_scores, self.generate_num)
                
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            
            metrics_res /= test_total
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            logging.info(f'Evaluation time for {dataset_name}/{task_name}: {elapsed_time:.2f} seconds')
            
            # Log timing to wandb
            if self.args.use_wandb:
                wandb.log({
                    f"{dataset_name}/{task_name}/eval_time": elapsed_time
                })
            
            for i in range(len(self.metrics)):
                logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
                
            if return_metrics:
                return metrics_res
    
    def test_dataset_task_filtered(self, testloader, return_metrics=False):
        """
        Run filtered evaluation on a test dataset one user at a time. Similar to test_dataset_task_filtered_batch,
        but processes users individually rather than in batches.
        
        This method:
        1. Takes one user at a time from the test set
        2. Gets their training history
        3. Creates a candidate set excluding items from their history
        4. Generates predictions only from the filtered candidates
        5. Evaluates metrics on the filtered predictions
        
        This is more memory efficient but slower than batch processing.
        
        Args:
            testloader: DataLoader containing test data with individual user histories and ground truth items.
                       Must have batch_size=1.
        """
        dataset_name = testloader.dataset.dataset
        task_name = testloader.dataset.task
        logging.info(f'testing filtered {dataset_name} dataset on {task_name} task')
        
        # Start timing
        start_time = time.time()
        
        test_total = 0
        with torch.no_grad():
            candidates = set(testloader.dataset.all_items)
            
            
            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                user_idx = int(batch[5][0])
                positive = testloader.dataset.positive[testloader.dataset.id2user[user_idx]]
                
                user_candidate = candidates - positive
                
                candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in user_candidate
                ]
                )
                prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
                
                prediction = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=30,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num,
                        num_return_sequences=self.generate_num,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]
                
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                
                rel_results = evaluate.rel_results(generated_sents, gold_sents, prediction_scores, self.generate_num)
                
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            
            metrics_res /= test_total
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            logging.info(f'Evaluation time for {dataset_name}/{task_name}: {elapsed_time:.2f} seconds')
            
            # Log timing to wandb
            if self.args.use_wandb:
                wandb.log({
                    f"{dataset_name}/{task_name}/eval_time": elapsed_time
                })
            
            for i in range(len(self.metrics)):
                logging.info(f'{self.metrics[i]}: {metrics_res[i]}')

            if return_metrics:
                return metrics_res
    
    def test_dataset_task(self, testloader, return_metrics=False):
        """
        Run standard (unfiltered) evaluation on a test dataset.
        """
        logging.info("="*50)
        dataset_name = testloader.dataset.dataset
        task_name = testloader.dataset.task
        logging.info(f'EVALUATION: {dataset_name} dataset on {task_name} task')
        
        # Start timing
        total_start = time.time()
        
        test_total = 0
        with torch.no_grad():
            # Time trie construction
            trie_start = time.time()
            candidates = testloader.dataset.all_items
            candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in candidates
                ]
            )
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
            trie_time = time.time() - trie_start
            logging.info(f"Trie construction took: {trie_time:.2f} seconds")
            
            metrics_res = np.array([0.0] * len(self.metrics))
            
            # Track cumulative times
            total_generate_time = 0
            total_decode_time = 0
            total_metric_time = 0
            
            for batch_idx, batch in enumerate(tqdm(testloader)):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                
                # Time model generation
                generate_start = time.time()
                prediction = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=30,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num,
                        num_return_sequences=self.generate_num,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                generate_time = time.time() - generate_start
                total_generate_time += generate_time
                
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]
                
                # Time tokenizer decoding
                decode_start = time.time()
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                decode_time = time.time() - decode_start
                total_decode_time += decode_time
                
                # Time metric calculation
                metric_start = time.time()
                rel_results = evaluate.rel_results(generated_sents, gold_sents, prediction_scores, self.generate_num)
                test_total += len(rel_results)
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                metric_time = time.time() - metric_start
                total_metric_time += metric_time
                
                # Log times for first few batches
                if batch_idx < 3:
                    logging.info(f"\nBatch {batch_idx} timing:")
                    logging.info(f"  Generation time: {generate_time:.2f}s")
                    logging.info(f"  Decode time: {decode_time:.2f}s")
                    logging.info(f"  Metric calc time: {metric_time:.2f}s")
                    logging.info(f"  Batch size: {len(input_ids)}")
            
            total_time = time.time() - total_start
            
            # Log overall timing breakdown
            logging.info("\nTiming Summary:")
            logging.info(f"Total evaluation time: {total_time:.2f}s")
            logging.info(f"Trie construction: {trie_time:.2f}s ({100*trie_time/total_time:.1f}%)")
            logging.info(f"Total generation time: {total_generate_time:.2f}s ({100*total_generate_time/total_time:.1f}%)")
            logging.info(f"Total decode time: {total_decode_time:.2f}s ({100*total_decode_time/total_time:.1f}%)")
            logging.info(f"Total metric calc time: {total_metric_time:.2f}s ({100*total_metric_time/total_time:.1f}%)")
            logging.info(f"Average time per batch:")
            logging.info(f"  Generation: {total_generate_time/len(testloader):.2f}s")
            logging.info(f"  Decode: {total_decode_time/len(testloader):.2f}s")
            logging.info(f"  Metric calc: {total_metric_time/len(testloader):.2f}s")
            
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            
            metrics_res /= test_total
            
            # Calculate elapsed time
            elapsed_time = time.time() - total_start
            logging.info(f'Evaluation time for {dataset_name}/{task_name}: {elapsed_time:.2f} seconds')
            
            # Log timing to wandb
            if self.args.use_wandb:
                wandb.log({
                    f"{dataset_name}/{task_name}/eval_time": elapsed_time
                })
            
            # Log results
            for i in range(len(self.metrics)):
                logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
                
            if return_metrics:
                return metrics_res
