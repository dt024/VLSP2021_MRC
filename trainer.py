# -*- coding: utf-8 -*-
import collections
import math
from datasets import load_metric
import numpy as np
import torch
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features, SquadExample
import os
import time
import datetime
import random
from transformers import BertForQuestionAnswering, AutoTokenizer, AutoConfig, AutoModel, BertModel, BertPreTrainedModel, BertConfig, RobertaConfig, RobertaModel, BertTokenizer, BertModel, RobertaTokenizer
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from official import nlp
import official.nlp.optimization
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
import logging
from tqdm import tqdm, trange
from utils import get_predictions, compute_metrics, asMinutes, timeSince, train_enc, valid_enc, test_enc

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, dev_answer=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_enc(train_dataset)
        self.eval_features, self.eval_dataset = valid_enc(dev_dataset)
        self.test_features, self.test_dataset = test_enc(test_dataset)
        self.epochs_stop = args.early_stop
        self.answers = dev_answers

        # self.config = BertConfig.from_pretrained(
        #     args.model_name_or_path,
        #     num_labels=self.num_labels,
        #     finetuning_task='VNLaw-BERT',
        #     output_hidden_states=True,
        #     output_attentions=True,
        # )
        self.model = BertForQuestionAnswering.from_pretrained(args.model_name_or_path)

        # self.model.resize_token_embeddings(len(tokenizer)) 

        # GPU or CPU
        self.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
      train_dataloader = DataLoader(
          self.train_dataset,
          shuffle=True,
          batch_size=self.args.train_batch_size,
          drop_last=True,
      )

      if self.args.max_steps > 0:
          t_total = self.args.max_steps
          self.args.num_train_epochs = (
              self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
          )
      else:
          t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

      # Prepare optimizer and schedule (linear warmup and decay)
      no_decay = ["bias", "LayerNorm.weight"]
      optimizer_grouped_parameters = [
          {
              "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
              "weight_decay": args.weight_decay,
          },
          {
              "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
              "weight_decay": 0.0,
          },
      ]
      optimizer = optim.AdamW(
          optimizer_grouped_parameters,
          lr=self.args.learning_rate,
          eps=self.args.adam_epsilon,
      )
      optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
      scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=self.args.warmup_steps,
          num_training_steps=t_total,
      )


      # Train!
      self.args.logging_steps = t_total // self.args.num_train_epochs
      self.args.save_steps    = t_total // self.args.num_train_epochs
      logger.info("***** Running training *****")
      logger.info("  Num examples = %d", len(self.train_dataset))
      logger.info("  Num Epochs = %d", self.args.num_train_epochs)
      logger.info("  Total train batch size = %d", self.args.train_batch_size)
      logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
      logger.info("  Total optimization steps = %d", t_total)
      logger.info("  Logging steps = %d", self.args.logging_steps)
      logger.info("  Save steps = %d", self.args.save_steps)

      global_step = 0
      tr_loss = 0.0
      max_val_score = -1
      early_stop = False
      epochs_no_improve = 0 

      # train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch") 
      for epoch_cnt in range(self.args.num_train_epochs):
          start = time.time()
          self.model.train()

          # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
          for step, batch in enumerate(train_dataloader):
              optimizer.zero_grad()
              batch[3] = batch[3].unsqueeze(1).to('cuda')
              batch[4] = batch[4].unsqueeze(1).to('cuda')
              batch = tuple(t.unsqueeze(0).to(self.device) if t.dim()==1 else t.to(self.device) for t in batch)  # GPU or CPU
              inputs = {
                  "input_ids": batch[0],
                  "attention_mask": batch[1],
                  "token_type_ids": batch[2],
                  "start_positions": batch[3],
                  "end_positions": batch[4],
              }
              outputs = self.model(**inputs)
              loss = outputs[0]

              if self.args.gradient_accumulation_steps > 1:
                  loss = loss / self.args.gradient_accumulation_steps

              loss.backward()

              tr_loss += loss.item()
              if (step + 1) % self.args.gradient_accumulation_steps == 0:
                  # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                  optimizer.step()
                  # scheduler.step()  # Update learning rate schedule

                  if step % 100 == 0 or step == (len(train_dataloader) - 1):
                      logger.info(
                          f"Epoch: [{epoch_cnt + 1}][{step}/{len(train_dataloader)}] "
                          f"Elapsed {timeSince(start, float(step + 1) / len(train_dataloader)):s} "
                          f"Loss: {tr_loss/(global_step+1):.4f} "
                      )
                  global_step += 1

                  if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                      res = self.evaluate("dev")
                      logger.info("Training total loss = %.4f", tr_loss/global_step)  
                      if res['f1']  > max_val_score:
                        max_val_score = res['f1']
                        final = res
                        epochs_no_improve = 0
                        self.save_model()
                      else:
                        epochs_no_improve += 1
                      
                      if epochs_no_improve == self.epochs_stop:
                        early_stop = True
                        logger.info(" Early Stopping!!!!!!")
                        logger.info("***** Final results *****")
                        for key in sorted(final.keys()):
                            logger.info("  {} = {:.4f}".format(key, final[key]))
                        break

                  # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                  #     self.save_model()

              if 0 < self.args.max_steps < global_step or early_stop==True:
                  epoch_iterator.close()
                  break
          
          if 0 < self.args.max_steps < global_step or early_stop==True:
              train_iterator.close()
              break

      return global_step, tr_loss / global_step

    def evaluating(self, mode, out_pred=False):
            # We use test dataset because semeval doesn't have dev dataset
            if mode == "test":
                features = self.test_features 
				datasets = self.test_dataset
            elif mode == "dev":
                features = self.eval_features
				datasets = self.eval_dataset
            else:
                raise Exception("Only dev and test dataset available")

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_start = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end = torch.tensor([f.end_position for f in features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start, all_end, all_example_index)

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batchsize)

            # Eval!
            logger.info("***** Running evaluation on %s dataset *****", mode)
            logger.info("  Num examples = %d", len(dataset))
            logger.info("  Batch size = %d", args.eval_batchsize)
            eval_loss = 0.0

            self.model.eval()
        
            all_results = []
            tmp_eval_loss = 0.0
            for batch in eval_dataloader:
                start = time.time()
                batch[3] = batch[3].unsqueeze(1).to('cuda')
                batch[4] = batch[4].unsqueeze(1).to('cuda')
                example_indexes = batch[5]
                batch = tuple(t.unsqueeze(0).to('cuda') if t.dim()==1 else t.to('cuda') for t in batch)  
                with torch.no_grad():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "start_positions": batch[3],
                        "end_positions": batch[4],
                    }
                    outputs = self.model(**inputs)
                    batch_start_logits, batch_end_logits = outputs[1],outputs[2]
                    # print(batch_start_logits)
                tmp_eval_loss += outputs[0]

                for i, example_index in enumerate(example_indexes):
                    start_logits = batch_start_logits[i].detach().cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                    eval_feature = features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    all_results.append(RawResult(unique_id=unique_id,
                                                start_logits=start_logits,
                                                end_logits=end_logits))

            predicts = get_predictions(datasets, features, all_results,
                                      self.args.n_best_size, self.args.max_answer_length,
                                      self.args.do_lower_case, self.args.verbose_logging,
                                      self.args.version_2_with_negative, self.args.null_score_diff_threshold)

            # return predicts
            # print(tmp_eval_loss)
			if mode == "test":
				return predicts
            result = compute_metrics(predicts, self.answers)
            eval_loss = tmp_eval_loss.item() / len(eval_dataloader)
            logger.info(
                        f"VAL: [{nb_eval_steps}/{len(eval_dataloader)}] "
                        f"Elapsed {timeSince(start, float(nb_eval_steps + 1) / len(eval_dataloader)):s} "
                        f"Loss: {eval_loss:.4f} "
                        )


            results = {"loss": eval_loss}
            # print(preds)
            # preds = np.argmax(preds, axis=1)
            # write_prediction(self.args, os.path.join(self.args.eval_dir, "proposed_answers.txt"), preds)

            
            results.update(result)

            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
              logger.info("  {} = {:.4f}".format(key, results[key]))

            if out_pred == True:
              return predicts
            return results
        

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.model_dir)
        # model_to_save = self.model
        # torch.save(model_to_save, os.path.join(self.args.model_dir, "model.pt"))

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        self.args = torch.load(os.path.join(self.args.model_dir, "training_args.bin"))
        self.config = BertConfig.from_pretrained(self.args.model_dir)

        self.model = Model.from_pretrained(self.args.model_dir, config=self.config, args=self.args)
        # self.model = torch.load(os.path.join(self.args.model_dir, "model.pt"))
        self.model.to(self.device)
        logger.info("***** Model Loaded *****")