from runner.SingleRunner import SingleRunner
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.distributed as dist
import logging
from tqdm import tqdm
from utils import utils
import torch
import utils.generation_trie as gt
import utils.evaluate as evaluate
from torch.utils.data.distributed import DistributedSampler
from data.TestDataset import TestDataset
from torch.utils.data import DataLoader
from processor.Collator import Collator, TestCollator
import time
import numpy as np
import random
import json
from collections import defaultdict
import wandb
import os

import pdb

class DistributedRunner(SingleRunner):
    
    def __init__(self, model, tokenizer, train_loader, valid_loader, device, args, rank):
        self.rank = rank
        super().__init__(model, tokenizer, train_loader, valid_loader, device, args)
        self.model = DDP(self.model, device_ids=[self.args.gpu], find_unused_parameters=True)
        
    def train(self):
        
        self.model.zero_grad()
        train_losses = []
        valid_losses = []
        best_epoch = -1
        if self.test_before_train > 0:
            self.test()
        
        for epoch in range(self.args.epochs):
            if self.rank == 0:
                logging.info(f"Start training for epoch {epoch+1}")
                
            dist.barrier()
            if self.regenerate_candidate:
                for ds in self.train_loader.dataset.datasets:
                    ds.generate_candidates()
                    ds.construct_sentence()
            elif self.reconstruct_data:
                for ds in self.train_loader.dataset.datasets:
                    ds.construct_sentence()
                    
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()
            
            self.model.train()
            losses = []
            
            for batch in tqdm(self.train_loader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                
                output = self.model.module(
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
                
                dist.barrier()
                
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                
                
                dist.all_reduce(loss.detach(), op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()
                
                dist.barrier()
                
                if self.rank == 0:
                    losses.append(loss.detach())
                
            if self.rank == 0:
                train_epoch_loss = sum(losses)/len(losses)
                train_losses.append(train_epoch_loss)
                logging.info(f"The average training loss for epoch {epoch+1} is {train_epoch_loss}")
                
            
        
            if self.valid_select > 0:
                if self.rank == 0:
                    logging.info(f"Start validation for epoch {epoch+1}")
                losses = []
                self.model.eval()
                with torch.no_grad():
                    if self.args.valid_prompt_sample > 0:
                        for ds in self.valid_loader.dataset.datasets:
                            ds.construct_sentence()
                    for batch in tqdm(self.valid_loader):
                        input_ids = batch[0].to(self.device)
                        attn = batch[1].to(self.device)
                        whole_input_ids = batch[2].to(self.device)
                        output_ids = batch[3].to(self.device)
                        output_attention = batch[4].to(self.device)

                        output = self.model.module(
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

                        dist.barrier()

                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss /= dist.get_world_size()

                        dist.barrier()

                        if self.rank == 0:
                            losses.append(loss)

                    if self.rank == 0:
                        valid_epoch_loss = sum(losses)/len(losses)
                        valid_losses.append(valid_epoch_loss)
                        logging.info(f"The average valid loss for epoch {epoch+1} is {valid_epoch_loss}")

                        if valid_epoch_loss == min(valid_losses):
                            logging.info(f"The minimal validation loss so far.")
                            best_epoch = epoch + 1
                            torch.save(self.model.module.state_dict(), self.args.model_path)
                            logging.info(f"Save the current model to {self.args.model_path}")
                
            if self.test_epoch > 0:
                if (epoch + 1) % self.test_epoch == 0:
                    self.model.eval()
                    self.test()
            
            dist.barrier()
        if self.valid_select > 0:
            if self.rank == 0:
                logging.info(f"The best validation at Epoch {best_epoch}")
        else:
            if self.rank == 0:
                torch.save(self.model.module.state_dict(), self.args.model_path)
                logging.info(f"Save the current model to {self.args.model_path}")
        
        return
    
    def get_testloader(self):
        """
        Create distributed test data loaders for each dataset and task combination.
        Supports loading multiple evaluation datasets.
        """
        self.testloaders = []
        
        # Get all datasets to evaluate on
        if hasattr(self.args, 'eval_datasets'):
            eval_datasets = self.args.eval_datasets.split(',')
        else:
            # Fallback to training datasets if eval_datasets not specified
            eval_datasets = self.args.datasets.split(',')
        
        tasks = self.args.tasks.split(',')
        
        if self.test_filtered > 0:
            collator = TestCollator(self.tokenizer)
        else:
            collator = Collator(self.tokenizer)
        
        if self.rank == 0:
            logging.info(f"Setting up evaluation for datasets: {eval_datasets}")
            logging.info(f"Tasks to evaluate: {tasks}")
        
        for dataset in eval_datasets:
            for task in tasks:
                if self.rank == 0:
                    logging.info(f"Creating test loader for {dataset} - {task}")
                testdata = TestDataset(self.args, dataset, task)
                test_sampler = DistributedSampler(testdata)
                testloader = DataLoader(
                    dataset=testdata,
                    sampler=test_sampler,
                    batch_size=self.args.eval_batch_size,
                    collate_fn=collator,
                    shuffle=False
                )
                self.testloaders.append(testloader)
    
    def test(self, path=None):
        """
        Run distributed evaluation on test datasets.
        """
        self.model.eval()
        if path:
            self.model.module.load_state_dict(torch.load(path, map_location=self.device))

        all_results = defaultdict(dict)
        
        # Initialize wandb if requested (only on rank 0)
        if self.rank == 0 and self.args.use_wandb:
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
                
            if self.rank == 0:
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

        if self.rank == 0:
            # Log results
            logging.info("\nEVALUATION RESULTS:")
            logging.info(json.dumps(all_results, indent=2))
            
            # Save results
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

        dist.barrier()
        return all_results if self.rank == 0 else None
    
    def test_dataset_task_filtered_batch(self, testloader, return_metrics=False):
        if self.rank == 0:
            logging.info(f'testing filtered {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0
        with torch.no_grad():
            candidates = set(testloader.dataset.all_items)
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
                user_idx = batch[5]
                
                
                
                prediction = self.model.module.generate(
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
                
                rel_results = evaluate.rel_results_filtered(testloader.dataset.positive, testloader.dataset.id2user, user_idx.detach().cpu().numpy(), \
                                                            self.generate_num+testloader.dataset.max_positive, \
                                                            generated_sents, gold_sents, prediction_scores, self.generate_num)
                
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)
            
            metrics_res /= test_total
            
            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
            
            if return_metrics:
                return metrics_res
    
    def test_dataset_task_filtered(self, testloader, return_metrics=False):
        if self.rank == 0:
            logging.info(f'testing filtered {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
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
                
                prediction = self.model.module.generate(
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
                
            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)
            
            metrics_res /= test_total
            
            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
            
            if return_metrics:
                return metrics_res
            
    def test_dataset_task(self, testloader, return_metrics=False):
        """
        Run standard (unfiltered) evaluation on a test dataset.
        """
        dataset_name = testloader.dataset.dataset
        task_name = testloader.dataset.task
        if self.rank == 0:
            logging.info("="*50)
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
            if self.rank == 0:
                logging.info(f"Trie construction took: {trie_time:.2f} seconds")
            
            # Track cumulative times
            total_generate_time = 0
            total_decode_time = 0
            total_metric_time = 0
            total_reduce_time = 0
            
            metrics_res = np.array([0.0] * len(self.metrics))
            for batch_idx, batch in enumerate(tqdm(testloader, disable=self.rank != 0)):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                
                # Time model generation
                generate_start = time.time()
                prediction = self.model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=50,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num,
                        num_return_sequences=self.generate_num,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                torch.cuda.synchronize()  # Ensure generation is complete
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
                
                # Log first few batches from rank 0
                if self.rank == 0 and batch_idx < 3:
                    logging.info(f"\nBatch {batch_idx} timing (rank 0):")
                    logging.info(f"  Generation time: {generate_time:.2f}s")
                    logging.info(f"  Decode time: {decode_time:.2f}s")
                    logging.info(f"  Metric calc time: {metric_time:.2f}s")
                    logging.info(f"  Batch size: {len(input_ids)}")
            
            # Time all_reduce operations
            reduce_start = time.time()
            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)
            reduce_time = time.time() - reduce_start
            total_reduce_time += reduce_time
            
            metrics_res /= test_total
            
            if self.rank == 0:
                total_time = time.time() - total_start
                
                # Log timing breakdown
                logging.info("\nTiming Summary (rank 0):")
                logging.info(f"Total evaluation time: {total_time:.2f}s")
                logging.info(f"Trie construction: {trie_time:.2f}s ({100*trie_time/total_time:.1f}%)")
                logging.info(f"Total generation time: {total_generate_time:.2f}s ({100*total_generate_time/total_time:.1f}%)")
                logging.info(f"Total decode time: {total_decode_time:.2f}s ({100*total_decode_time/total_time:.1f}%)")
                logging.info(f"Total metric calc time: {total_metric_time:.2f}s ({100*total_metric_time/total_time:.1f}%)")
                logging.info(f"Total reduce time: {total_reduce_time:.2f}s ({100*total_reduce_time/total_time:.1f}%)")
                logging.info(f"Average time per batch:")
                logging.info(f"  Generation: {total_generate_time/len(testloader):.2f}s")
                logging.info(f"  Decode: {total_decode_time/len(testloader):.2f}s")
                logging.info(f"  Metric calc: {total_metric_time/len(testloader):.2f}s")
                logging.info(f"  Reduce ops: {total_reduce_time/len(testloader):.2f}s")
                
                # Log to wandb
                if self.args.use_wandb:
                    wandb.log({
                        f"{dataset_name}/{task_name}/eval_time": total_time,
                        f"{dataset_name}/{task_name}/trie_time": trie_time,
                        f"{dataset_name}/{task_name}/generation_time": total_generate_time,
                        f"{dataset_name}/{task_name}/decode_time": total_decode_time,
                        f"{dataset_name}/{task_name}/metric_time": total_metric_time,
                        f"{dataset_name}/{task_name}/reduce_time": total_reduce_time,
                        f"{dataset_name}/{task_name}/avg_batch_time": total_time/len(testloader)
                    })
            
            if return_metrics:
                return metrics_res
                    
