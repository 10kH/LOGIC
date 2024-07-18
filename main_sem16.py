import os
import argparse
import logging

import pandas as pd
import numpy as np
import torch.optim

from sklearn.metrics import f1_score, precision_score, recall_score

from dataset import *
from math import ceil
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from models import BartForConditionalGeneration
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        default='facebook/bart-base',
        type=str

    )

    parser.add_argument(
        '--pad_token_id',
        default=1,
        type=int
    )

    parser.add_argument(
        '--epoch',
        default=30,
        type=int
    )


    parser.add_argument(
        '--num_warmup_ratio',
        default=0.1,
        type=float
    )

    parser.add_argument(
        '--seed',
        default=428,
        type=int
    )

    parser.add_argument(
        '--lr',
        default=5e-6,
        type=float
    )

    parser.add_argument(
        '--batch_size',
        default=8,
        type=int
    )

    parser.add_argument(
        '--accumulate_step',
        default=4,
        type=int
    )

    parser.add_argument(
        '--enc_max_length',
        default=524,
        type=int
    )

    parser.add_argument(
        '--dec_max_length',
        default=1024,
        type=int
    )

    parser.add_argument(
        '--n_sample',
        default=-1,
        type=int
    )

    parser.add_argument(
        '--reasoning_generation_alpha',
        default=3,
        type=float
    )

    parser.add_argument(
        '--unlikelihood_training_alpha',
        default=0.5,
        type=float
    )

    parser.add_argument(
        '--topic_prediction_alpha',
        default=0.5,
        type=float
    )

    parser.add_argument(
        '--is_long',
        action='store_true',
        help="Adding reasoning long.")

    parser.add_argument(
        '--is_short',
        action='store_true',
        help="Adding reasoning short.")

    parser.add_argument(
        '--with_unlikelihood_training',
        action='store_true',
        help="Adding unlikelihood training.")


    parser.add_argument(
        '--with_topic_prediction',
        action='store_true',
        help="Adding topic generation data.")

    parser.add_argument(
        '--with_reason',
        action='store_true',
        help="Adding reason."
    )

    parser.add_argument(
        '--with_wiki',
        action='store_true',
        help="Adding wiki."
    )

    parser.add_argument(
        '--target_knowledge_made_by_llm',
        action='store_true',
        help="Adding LLM target knowledge."
    )

    parser.add_argument(
        '--with_topic',
        action='store_true',
        help="Adding topic in output."
    )

    parser.add_argument(
        '--with_tracking',
        action='store_true',
        help="Adding tracking."
    )

    parser.add_argument(
        '--stance_is_in_front',
        action='store_true',
        help="Deciding whether Stance comes before or after"
    )

    parser.add_argument(
        '--device',
        default="cuda",
        type=str
    )

    parser.add_argument(
        '--train_path',
        default="data/sem16t6_vast_df.csv",
        type=str
    )
    # ['hillary clinton', 'legalization of abortion', 'atheism', 'climate change is a real concern', 'feminist movement', 'donald trump']
    #donald trump
    #legalization of abortion
    #hillary clinton
    #feminist movement
    parser.add_argument(
        '--test_target',
        default="donald trump",
        type=str
    )

    parser.add_argument(
        '--output_dir',
        default="../save_dir/Testing",
        type=str
    )

    args = parser.parse_args()

    if not os.path.exists(f"../log"):
        # 폴더가 없으면 생성
        os.makedirs(f"../log")

    # args.output_dir = args.output_dir + f"_{args.batch_size*args.accumulate_step}Batch_size/"

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S",
                        filemode="w",
                        # filename=f"../log/add_reason_{args.n_sample}_ts_{args.batch_size*args.accumulate_step}Batch_size.log",
                        level=logging.INFO)

    return args

# self.label_mapping_talbe = {
#             0: 'negative',
#             1: 'positive',
#             2: 'neutral'
#         }


def map_gen_to_prediction(decoded_text):
    decoded_text = decoded_text.split(" ")
    # Default is neutral
    stance = 2
    for i in range(len(decoded_text)-2):
        if decoded_text[i] == 'Stance' and decoded_text[i + 1] == 'is':
            stance_text = decoded_text[i+2].strip(".")
            stance = 0 if stance_text == 'negative' else 1 if stance_text == 'positive' else 2
    return stance
def evaluate(model, dataloader, device, tokenizer, args):
    model.eval()
    all_logits, all_labels, all_data_type_marks = [], [], []
    generated_labels = []
    ground_truth_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            kwargs = dict()

            kwargs["input_ids"] = batch["input_ids"].to(device)
            kwargs["attention_mask"] = batch["attention_mask"].to(device)
            kwargs["num_beams"] = 1
            kwargs["do_sample"] = False
            # kwargs["max_length"] = args.dec_max_length
            kwargs["max_length"] = 50
            labels = batch["labels"].to(device)

            generated_tokens = model.generate(**kwargs)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            pred_labels = [map_gen_to_prediction(x) for x in decoded_preds]
            true_labels = [map_gen_to_prediction(x) for x in decoded_labels]
            generated_labels += decoded_preds
            ground_truth_labels += decoded_labels
            all_labels += true_labels
            predicted_labels += pred_labels
            all_data_type_marks += batch["stances"]

    predicted_labels = np.array(predicted_labels)
    all_labels = np.array(all_labels)
    # all_data_type_marks = torch.cat(all_data_type_marks, dim=0).cpu().numpy()

    metrics = dict()
    # metrics["f1"] = f1_score(all_labels, predicted_labels, average='macro')
    # metrics["precision"] = precision_score(all_labels, predicted_labels, average='macro')
    # metrics["recall"] = recall_score(all_labels, predicted_labels, average='macro')

    # for mark in ['negative', 'positive']:
    #     subset_predicted_labels = []
    #     subset_true_labels = []
    #     for i in range(len(all_labels)):
    #         if all_data_type_marks[i] == mark:
    #             subset_predicted_labels.append(predicted_labels[i])
    #             subset_true_labels.append(all_labels[i])
    #     metrics["f1_" + str(mark)] = f1_score(subset_true_labels, subset_predicted_labels)


    subset_predicted_labels = []
    subset_true_labels = []
    for i in range(len(all_labels)):
        if all_data_type_marks[i] in ['negative', 'positive']:
            subset_predicted_labels.append(predicted_labels[i])
            subset_true_labels.append(all_labels[i])
    metrics["f1_avg"] = round(f1_score(all_labels, predicted_labels, average='macro', labels=[0,1]), 3)
    metrics["f1_neg"] = round(f1_score(all_labels, predicted_labels, average='macro', labels=[0]), 3)
    metrics["f1_pos"] = round(f1_score(all_labels, predicted_labels, average='macro', labels=[1]), 3)
        # metrics["precision_" + str(mark)] = precision_score(subset_true_labels, subset_predicted_labels,
        #                                                     average='macro')
        # metrics["recall" + str(mark)] = recall_score(subset_true_labels, subset_predicted_labels, average='macro')

    return metrics, generated_labels, ground_truth_labels
def train(model, train_dataloader, train_dataloader_topic_prediction, train_dataloader_unlikelihood, train_dataloader_topic_unlikelihood, train_dataloader_short, train_dataloader_long , dev_dataloader, device, tokenizer, args):
    weight_decay = 0
    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in parameters_to_optimize if
    #                 not any(nd in n for nd in no_decay)],
    #      'weight_decay': weight_decay},
    #     {'params': [p for n, p
    #     in parameters_to_optimize if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0}
    # ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)

    num_training_steps_per_epoch = ceil(len(train_dataloader.dataset) / (args.batch_size * args.accumulate_step))
    num_traing_steps = num_training_steps_per_epoch * args.epoch

    num_warmup_steps = ceil(args.num_warmup_ratio * num_traing_steps)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps= num_warmup_steps,
        num_training_steps=num_traing_steps,
    )

    model, optimizer, train_dataloader, dev_dataloader, lr_scheduler = accelerator.prepare(model,
                                                                           optimizer,
                                                                           train_dataloader,
                                                                           dev_dataloader,
                                                                           lr_scheduler)
    #train_dataloader_topic_prediction, train_dataloader_unlikelihood, train_dataloader_shuffling, train_dataloader_short, train_dataloader_long
    if train_dataloader_topic_prediction is not None:
        train_dataloader_topic_prediction = accelerator.prepare(train_dataloader_topic_prediction)
        topic_prediction_iter = iter(train_dataloader_topic_prediction)
    if train_dataloader_unlikelihood is not None:
        train_dataloader_unlikelihood = accelerator.prepare(train_dataloader_unlikelihood)
        unlikelihood_iter = iter(train_dataloader_unlikelihood)
    if train_dataloader_topic_unlikelihood is not None:
        train_dataloader_topic_unlikelihood = accelerator.prepare(train_dataloader_topic_unlikelihood)
        topic_unlikelihood_iter = iter(train_dataloader_topic_unlikelihood)
    if train_dataloader_short is not None:
        train_dataloader_short = accelerator.prepare(train_dataloader_short)
        short_iter = iter(train_dataloader_short)
    if train_dataloader_long is not None:
        train_dataloader_long = accelerator.prepare(train_dataloader_long)
        long_iter = iter(train_dataloader_long)

    best_f1 = 0.

    for epoch in range(1, args.epoch+1):
        model.train()
        logging.info(f"***** {epoch} Epoch Start ***** \n")

        for step, batch in tqdm(enumerate(train_dataloader),
                             desc=f'Running train for epoch {epoch}',
                             total=len(train_dataloader)):
            # input_ids, attention_mask, decoder_input_ids, labels, _ = batch
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # input_ids = input_ids.to(args.device)
            # attention_mask = attention_mask.to(args.device)
            # labels = labels.to(args.device)

            labels[labels == args.pad_token_id] = -100

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,)

            loss = outputs.loss
            loss /= args.accumulate_step
            # loss.backward()
            accelerator.backward(loss)

            if train_dataloader_topic_prediction is not None:
                try:
                    topic_prediciton_batch = next(topic_prediction_iter)
                except StopIteration:
                    topic_prediction_iter = iter(train_dataloader_topic_prediction)
                    topic_prediciton_batch = next(topic_prediction_iter)

                input_ids = topic_prediciton_batch['input_ids']
                attention_mask = topic_prediciton_batch['attention_mask']
                labels = topic_prediciton_batch['labels']

                labels[labels == args.pad_token_id] = -100

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels, )

                loss = outputs.loss
                loss /= args.accumulate_step
                accelerator.backward(loss)
            if train_dataloader_unlikelihood is not None:
                try:
                    unlikelihood_batch = next(unlikelihood_iter)
                except StopIteration:
                    unlikelihood_iter = iter(train_dataloader_unlikelihood)
                    unlikelihood_batch = next(unlikelihood_iter)

                input_ids = unlikelihood_batch['input_ids']
                attention_mask = unlikelihood_batch['attention_mask']
                negative_loss_mask = unlikelihood_batch['negative_loss_mask']
                labels = unlikelihood_batch['labels']

                labels[labels == args.pad_token_id] = -100

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                neg_loss_mask=negative_loss_mask,
                                is_neg=True)

                loss = outputs.loss
                loss /= args.accumulate_step
                accelerator.backward(loss * 0.5)

            if train_dataloader_short is not None:
                try:
                    short_batch = next(short_iter)
                except StopIteration:
                    short_iter = iter(train_dataloader_short)
                    short_batch = next(short_iter)

                input_ids = short_batch['input_ids']
                attention_mask = short_batch['attention_mask']
                labels = short_batch['labels']

                labels[labels == args.pad_token_id] = -100

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels, )

                loss = outputs.loss
                loss /= args.accumulate_step
                accelerator.backward(loss * args.reasoning_generation_alpha)

            if train_dataloader_long is not None:
                try:
                    long_batch = next(long_iter)
                except StopIteration:
                    long_iter = iter(train_dataloader_long)
                    long_batch = next(long_iter)

                input_ids = long_batch['input_ids']
                attention_mask = long_batch['attention_mask']
                labels = long_batch['labels']

                labels[labels == args.pad_token_id] = -100

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels, )

                loss = outputs.loss
                loss /= args.accumulate_step
                accelerator.backward(loss * args.reasoning_generation_alpha)

            if args.with_tracking:
                accelerator.log({"training_loss": loss.item()})
                accelerator.log({'learning_rate_1': lr_scheduler.get_last_lr()[0]})



            if (step + 1) % args.accumulate_step == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

        dev_metrics, generated_labels, ground_truth_labels = evaluate(model, dev_dataloader, device, tokenizer, args)

        with open(f"../log/results_{epoch}_ts.txt", "w") as f:
            for i in range(len(generated_labels)):
                f.write(f"{i + 1}.\nLabel : {ground_truth_labels[i]}\nPreidct: {generated_labels[i]}\n\n")

        log_text = ""
        for k in dev_metrics:
            log_text += k + ":" + str(dev_metrics[k]) + ","
        logging.info(log_text)

        if dev_metrics["f1_avg"] > best_f1:
            logging.info(f"New best, dev_f1_avg={dev_metrics['f1_avg']} > best_f1_avg={best_f1}")
            best_f1 = dev_metrics["f1_avg"]
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            model.save_pretrained(args.output_dir)


if __name__ == '__main__':
    args = parse_args()
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logging.info(f"Set random seed to {seed}")


    set_random_seed(args.seed)

    ###################################################################################
    # accelerator 선언
    if not args.with_tracking:
        accelerator = Accelerator()
    else:
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers("TRUSS")
    ###################################################################################

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(["<stance>", "<topic>", "<reasoning_short>", "<reasoning_long>"])

    ###
    args.pad_token_id = tokenizer.pad_token_id
    ##

    config = AutoConfig.from_pretrained(args.model_name)

    logging.info(f"Target -> {args.test_target}")
    data = pd.read_csv(args.train_path)
    data_train = data[data['topic_str'] != args.test_target]
    data_test = data[data['topic_str'] == args.test_target]

    # if args.n_sample > 0:
    #     data = data.sample(args.n_sample)
    #     logging.info(f"**** measure the distribution of the labels ****\n{data['label'].value_counts()}")

    train_dataset = Sem16Dataset(data=data_train, tokenizer=tokenizer, is_train=True, train_mode="stance_prediction", args=args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    train_dataloader_topic_prediction = None
    train_dataloader_unlikelihood = None
    train_dataloader_topic_unlikelihood = None
    train_dataloader_shuffling = None
    train_dataloader_short = None
    train_dataloader_long = None

    if args.with_topic_prediction:
        train_dataset_topic_prediction = Sem16Dataset(data=data_train, tokenizer=tokenizer, is_train=True,
                                                      train_mode="topic_prediction",
                                                      args=args)
        train_dataloader_topic_prediction = DataLoader(train_dataset_topic_prediction, batch_size=args.batch_size,
                                                       shuffle=True)

    if args.with_unlikelihood_training:
        train_dataset_unlikelihood = Sem16Dataset(data=data_train, tokenizer=tokenizer, is_train=True,
                                                  train_mode="unlikelihood_training",
                                                  args=args)
        train_dataloader_unlikelihood = DataLoader(train_dataset_unlikelihood, batch_size=args.batch_size,
                                                   shuffle=True)

    if args.is_short:
        train_dataset_short = Sem16ReasoningDataset(data_train, tokenizer=tokenizer, args=args, is_long=False)
        train_dataloader_short = DataLoader(train_dataset_short, batch_size=args.batch_size, shuffle=True)

    if args.is_long:
        train_dataset_long = Sem16ReasoningDataset(data_train, tokenizer=tokenizer, args=args, is_long=True)
        train_dataloader_long = DataLoader(train_dataset_long, batch_size=args.batch_size, shuffle=True)

    test_dataset = Sem16Dataset(data=data_test, tokenizer=tokenizer, args=args, is_train=False, train_mode="stance_prediction")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)


    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    train(
            model = model,
            train_dataloader=train_dataloader,
            train_dataloader_topic_prediction=train_dataloader_topic_prediction,
            train_dataloader_unlikelihood=train_dataloader_unlikelihood,
            train_dataloader_topic_unlikelihood=train_dataloader_topic_unlikelihood,
            train_dataloader_short=train_dataloader_short,
            train_dataloader_long=train_dataloader_long,
            dev_dataloader=test_dataloader,
            device=args.device,
            tokenizer=tokenizer,
            args=args
    )
    model = BartForConditionalGeneration.from_pretrained(args.output_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    test_metrics, generated_labels, ground_truth_labels = evaluate(model, test_dataloader, args.device, tokenizer, args)

    log_text = "Test results: "
    for k in test_metrics:
        log_text += k + ":" + str(test_metrics[k]) + ","
    logging.info(log_text)

    # python main.py --batch_size 1 --accumulate_step 1 --with_topic --with_reason

    # if args.with_tracking:
    #     accelerator.end_training()
    #
    # for batch in train_dataloader_topic_unlikelihood:
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     labels = batch['labels']
    #     print("Input Ids :")
    #     print(tokenizer.batch_decode(input_ids))
    #     print("\n Output Ids :")
    #     print(tokenizer.batch_decode(labels))
    #     print("\n Negative_loss_mask :")
    #     print(batch['negative_loss_mask'])
    #     print("===="*10)
    #     break
    # for batch in train_dataloader_unlikelihood:
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     labels = batch['labels']
    #     print("Input Ids :")
    #     print(tokenizer.batch_decode(input_ids))
    #     print("\n Output Ids :")
    #     print(tokenizer.batch_decode(labels))
    #     print("\n Negative_loss_mask :")
    #     print(batch['negative_loss_mask'])
    #     print("===="*10)
    #     break

    # for batch in test_dataloader:
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     labels = batch['labels']
    #     print("Input Ids :")
    #     print(tokenizer.batch_decode(input_ids))
    #     print("\n Output Ids :")
    #     print(tokenizer.batch_decode(labels))
    #     print("===="*10)
    #     break

    #
    # for batch in train_dataloader_topic_prediction:
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     labels = batch['labels']
    #     print("Input Ids :")
    #     print(tokenizer.batch_decode(input_ids))
    #     print("\n Output Ids :")
    #     print(tokenizer.batch_decode(labels))
    #     print("===="*10)
    #     break
    #
    # for batch in train_dataloader_unlikelihood:
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     labels = batch['labels']
    #     print("Input Ids :")
    #     print(tokenizer.batch_decode(input_ids))
    #     print("\n Output Ids :")
    #     print(tokenizer.batch_decode(labels))
    #     print("===="*10)
    #     break
    #
    # for batch in train_dataloader_short:
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     labels = batch['labels']
    #     print("Input Ids :")
    #     print(tokenizer.batch_decode(input_ids))
    #     print("\n Output Ids :")
    #     print(tokenizer.batch_decode(labels))
    #     print("====" * 10)
    #     break
    #
    # for batch in train_dataloader_long:
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     labels = batch['labels']
    #     print("Input Ids :")
    #     print(tokenizer.batch_decode(input_ids))
    #     print("\n Output Ids :")
    #     print(tokenizer.batch_decode(labels))
    #     print("===="*10)
    #     break
    # for batch in train_dataloader_shuffling:
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     labels = batch['labels']
    #     print("Input Ids :")
    #     print(tokenizer.batch_decode(input_ids))
    #     print("\n Output Ids :")
    #     print(tokenizer.batch_decode(labels))
    #     print("===="*10)
    #     break
    #
    #     if args.with_reasoning_generation:
    #         print("Reasoning Generation")
    #         input_ids = batch['input_ids_rg']
    #         labels = batch['labels_rg']
    #         print("Input Ids :")
    #         print(tokenizer.batch_decode(input_ids))
    #         print("\n Output Ids :")
    #         print(tokenizer.batch_decode(labels))
    #         print("====" * 10)
    #
    #     break

    #     if args.with_unlikelihood_training:
    #         print("Unlikelihood Training")
    #         input_ids = batch['input_ids_ut']
    #         labels = batch['labels_ut']
    #         print("Input Ids :")
    #         print(tokenizer.batch_decode(input_ids))
    #         print("\n Output Ids :")
    #         print(tokenizer.batch_decode(labels))
    #         # print("\n labels :")
    #         # print(batch['labels_ut'].shape)
    #         # print("\n Negative Loss Mask :")
    #         # print(batch['negative_loss_mask'].shape)
    #
    #         # for num, neg_mask in enumerate(batch['negative_loss_mask']):
    #         #     for idx, neg in enumerate(neg_mask):
    #         #         if neg > 0:
    #         #             print(tokenizer.convert_ids_to_tokens(batch['labels_ut'][num][idx].item()))
    #
    #         print("====" * 10)
    #
    #     if args.with_shuffling:
    #         print("Shuffling")
    #         input_ids = batch['input_ids_sh']
    #         labels = batch['labels_sh']
    #         print("Input Ids :")
    #         print(tokenizer.batch_decode(input_ids))
    #         print("\n Output Ids :")
    #         print(tokenizer.batch_decode(labels))
    #         print("====" * 10)
    #
    #     if args.with_topic_prediction:
    #         print("Topic Prediction")
    #         input_ids = batch['input_ids_tp']
    #         labels = batch['labels_tp']
    #         print("Input Ids :")
    #         print(tokenizer.batch_decode(input_ids))
    #         print("\n Output Ids :")
    #         print(tokenizer.batch_decode(labels))
    #         print("====" * 10)
    #
    #
    #     break

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
