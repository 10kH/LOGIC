import torch
import random
import pickle

from few_shot_examples import *
from torch.utils.data import Dataset, DataLoader


class VastDataset(Dataset):
    def __init__(self, data, tokenizer, args, train_mode, is_train=True, wiki_path=False):
        super().__init__()

        self.data = data

        self.enc_max_length = args.enc_max_length
        self.dec_max_length = args.dec_max_length

        if args.with_topic_str_chatgpt:
            self.topic_str_chatgpt = pickle.load(open("data/topic_str_chatgpt.pkl", "rb"))
            self.enc_max_length = 400

        # with_new_topic은 LLM을 통해서 외부 지식을 대체한 것을 말한다.
        # 여기서 Target으로 new_topic을 준 것이다.
        if args.with_new_topic:
            self.new_topic_chatgpt = pickle.load(open("data/new_topic_chatgpt.pkl", "rb"))
            self.enc_max_length = 400

        # with_tiki는 외부 지식으로 단순히 위키피디아 지식을 준 것이다.
        self.wiki_path = args.with_wiki
        if self.wiki_path:
            self.wiki_dict = pickle.load(open("data/wiki_dict.pkl", "rb"))
            self.enc_max_length = 512

        self.tokenizer = tokenizer

        self.is_train = is_train
        self.args = args
        self.train_mode = train_mode

        self.label_mapping_talbe = {
            0: 'negative',
            1: 'positive',
            2: 'neutral'
        }

        self.gen_datas = dict()
        self.preprocess_data()

    def generate_output_template(self, topic, reason, stance, is_train, train_mode, args):
        if not is_train:
            return f"Topic is {topic}. Stance is {stance}."

        offset_begin = None
        offset_end = None

        if train_mode == "stance_prediction":
            output_template = f"Topic is {topic}. Stance is {stance}."
        elif train_mode == "unlikelihood_training":
            # output_template = f"The reasons are as follows. {reason} so, Stance is "
            output_template = f"Topic is {topic}. Stance is "

            offset_begin = len(output_template)
            output_template += f"{stance}"
            offset_end = len(output_template)
            output_template += f"."
        elif train_mode == "topic_unlikelihood_training":
            # output_template = f"The reasons are as follows. {reason} so, Stance is "
            output_template = f"Topic is "
            offset_begin = len(output_template)
            output_template += f"{topic}"
            offset_end = len(output_template)
            output_template += f". Stance is {stance}."
        elif train_mode == "shuffling":
            output_template = f"Topic is {topic}. Stance is {stance}."
        elif train_mode == "topic_prediction":
            output_template = f"Topic is {topic}. Stance is {stance}."
        elif train_mode == "relation_prediction":
            pass
        else:
            raise ValueError(f"There is no {train_mode} in abblation_study.")

        return output_template, (offset_begin, offset_end)

    def preprocess_data(self):
        self.posts = self.data['post'].tolist()
        self.topics = self.data['topic_str'].tolist()
        self.topic_list = self.data['topic_str'].unique().tolist()
        self.labels = self.data['label'].tolist()
        self.labels = [self.label_mapping_talbe[x] for x in self.labels]
        self.new_topics = self.data['new_topic'].tolist()
        self.data_type_marks = self.data['seen?'].tolist()

        # if (self.args.is_long or self.args.is_short) and self.is_train:
        #     self.reasons = self.data['ChatGPT'].tolist()

        self.gen_datas['input_ids'] = list()
        self.gen_datas['decoder_input_ids'] = list()
        self.gen_datas['labels'] = list()
        self.gen_datas['attention_mask'] = list()
        self.gen_datas['data_type_mark'] = torch.LongTensor(self.data_type_marks)
        self.gen_datas['negative_loss_mask'] = list()

        self.max_length_in_data = 0

        for i in range(len(self.labels)):
            if not self.is_train:
                input_template = f"Topic is {self.topics[i]}. Stance is <stance>. </s></s>{self.posts[i]}"

                if self.args.with_topic_str_chatgpt:
                    input_template += f"</s></s>" + self.topic_str_chatgpt[self.topics[i]]

                if self.args.with_new_topic:
                    input_template += f"</s></s>" + self.new_topic_chatgpt[self.new_topics[i]]

                if self.wiki_path:
                    input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

                output_template = self.generate_output_template(topic=self.topics[i],
                                                                reason=None,
                                                                stance=self.labels[i],
                                                                is_train=False,
                                                                train_mode=self.train_mode,
                                                                args=self.args)

                input_text_tok = self.tokenizer(input_template, padding="max_length", max_length=self.enc_max_length,
                                                truncation=True)
                output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                 max_length=self.dec_max_length - 800,
                                                 truncation=True, return_offsets_mapping=True)

                self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
                self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
                self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))
                continue

            if self.train_mode == "stance_prediction":
                input_template = f"Topic is {self.topics[i]}. Stance is <stance>. </s></s>{self.posts[i]}"

                if self.args.with_topic_str_chatgpt:
                    input_template += f"</s></s>" + self.topic_str_chatgpt[self.topics[i]]

                if self.args.with_new_topic:
                    input_template += f"</s></s>" + self.new_topic_chatgpt[self.new_topics[i]]

                if self.wiki_path:
                    input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

                output_template, _ = self.generate_output_template(topic=self.topics[i],
                                                                   reason=None,
                                                                   stance=self.labels[i],
                                                                   is_train=self.is_train,
                                                                   train_mode=self.train_mode,
                                                                   args=self.args)

                input_text_tok = self.tokenizer(input_template, padding="max_length", max_length=self.enc_max_length,
                                                truncation=True)
                output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                 max_length=self.dec_max_length - 800,
                                                 truncation=True, return_offsets_mapping=True)

                self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
                self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
                self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))
            elif self.train_mode == "topic_prediction":
                input_template = f"Topic is <topic>. Stance is {self.labels[i]}. </s></s>{self.posts[i]}"

                if self.args.with_topic_str_chatgpt:
                    input_template += f"</s></s>" + self.topic_str_chatgpt[self.topics[i]]

                if self.args.with_new_topic:
                    input_template += f"</s></s>" + self.new_topic_chatgpt[self.new_topics[i]]

                if self.wiki_path:
                    input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

                output_template, _ = self.generate_output_template(topic=self.topics[i],
                                                                   reason=None,
                                                                   stance=self.labels[i],
                                                                   is_train=self.is_train,
                                                                   train_mode=self.train_mode,
                                                                   args=self.args)
                input_text_tok = self.tokenizer(input_template, padding="max_length", max_length=self.enc_max_length,
                                                truncation=True)
                output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                 max_length=self.dec_max_length - 800,
                                                 truncation=True, return_offsets_mapping=True)

                self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
                self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
                self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))
            elif self.train_mode == "shuffling":
                for stance in ['negative', 'positive', 'neutral']:
                    input_template = f"Topic is {self.topics[i]}. Stance is {stance}. </s></s>{self.posts[i]}"

                    if self.args.with_topic_str_chatgpt:
                        input_template += f"</s></s>" + self.topic_str_chatgpt[self.topics[i]]

                    if self.args.with_new_topic:
                        input_template += f"</s></s>" + self.new_topic_chatgpt[self.new_topics[i]]

                    if self.wiki_path:
                        input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

                    output_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                            reason=None,
                                                                            stance=self.labels[i],
                                                                            is_train=self.is_train,
                                                                            train_mode=self.train_mode,
                                                                            args=self.args)

                    input_text_tok = self.tokenizer(input_template, padding="max_length",
                                                    max_length=self.enc_max_length,
                                                    truncation=True)
                    output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                     max_length=self.dec_max_length - 800,
                                                     truncation=True, return_offsets_mapping=True)

                    self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
                    self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
                    self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))
            elif self.train_mode == "unlikelihood_training":
                input_template = f"Topic is {self.topics[i]}. Stance is <stance>. </s></s>{self.posts[i]}"

                if self.args.with_topic_str_chatgpt:
                    input_template += f"</s></s>" + self.topic_str_chatgpt[self.topics[i]]

                if self.args.with_new_topic:
                    input_template += f"</s></s>" + self.new_topic_chatgpt[self.new_topics[i]]

                if self.wiki_path:
                    input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

                fake_labels = ['negative', 'positive', 'neutral']
                fake_labels.remove(self.labels[i])

                for stance in fake_labels:
                    output_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                            reason=None,
                                                                            stance=stance,
                                                                            is_train=self.is_train,
                                                                            train_mode=self.train_mode,
                                                                            args=self.args)

                    input_text_tok = self.tokenizer(input_template, padding="max_length",
                                                    max_length=self.enc_max_length,
                                                    truncation=True)
                    output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                     max_length=self.dec_max_length - 800,
                                                     truncation=True, return_offsets_mapping=True)

                    neg_loss_mask = torch.zeros_like(torch.Tensor(output_text_tok['attention_mask']))

                    for j in range(len(output_text_tok["offset_mapping"])):
                        if (output_text_tok["offset_mapping"][j][0] >= offset[0]) and \
                                (output_text_tok["offset_mapping"][j][1] <= offset[1]):
                            neg_loss_mask[j] = 1.0

                    self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
                    self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
                    self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))
                    self.gen_datas["negative_loss_mask"].append(neg_loss_mask)
            elif self.train_mode == "topic_unlikelihood_training":
                input_template = f"Topic is <topic>. Stance is {self.labels[i]}. </s></s>{self.posts[i]}"

                if self.args.with_topic_str_chatgpt:
                    input_template += f"</s></s>" + self.topic_str_chatgpt[self.topics[i]]

                if self.args.with_new_topic:
                    input_template += f"</s></s>" + self.new_topic_chatgpt[self.new_topics[i]]

                if self.wiki_path:
                    input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

                fake_labels = self.topic_list[:]
                random.shuffle(fake_labels)

                try:
                    fake_labels.remove(self.new_topics[i])
                except Exception:
                    pass

                # for topic in fake_labels[:self.args.epoch]:
                for topic in fake_labels[:self.args.epoch]:
                    output_template, offset = self.generate_output_template(topic=topic,
                                                                            reason=None,
                                                                            stance=self.labels[i],
                                                                            is_train=self.is_train,
                                                                            train_mode=self.train_mode,
                                                                            args=self.args)

                    input_text_tok = self.tokenizer(input_template, padding="max_length",
                                                    max_length=self.enc_max_length,
                                                    truncation=True)
                    output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                     max_length=self.dec_max_length - 800,
                                                     truncation=True, return_offsets_mapping=True)

                    neg_loss_mask = torch.zeros_like(torch.Tensor(output_text_tok['attention_mask']))

                    for j in range(len(output_text_tok["offset_mapping"])):
                        if (output_text_tok["offset_mapping"][j][0] >= offset[0]) and \
                                (output_text_tok["offset_mapping"][j][1] <= offset[1]):
                            neg_loss_mask[j] = 1.0

                    self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
                    self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
                    self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))
                    self.gen_datas["negative_loss_mask"].append(neg_loss_mask)

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        outputs = dict()

        outputs['input_ids'] = self.gen_datas['input_ids'][idx]
        outputs['attention_mask'] = self.gen_datas['attention_mask'][idx]
        outputs['labels'] = self.gen_datas['labels'][idx]
        outputs['data_type_mark'] = self.gen_datas['data_type_mark'][idx]

        if self.train_mode == "unlikelihood_training" or self.train_mode == "topic_unlikelihood_training":
            outputs['negative_loss_mask'] = self.gen_datas['negative_loss_mask'][idx]

        return outputs

class Sem16Dataset(Dataset):
    def __init__(self, data, tokenizer, args, train_mode, is_train=True, wiki_path=False):
        super().__init__()

        self.data = data

        self.enc_max_length = args.enc_max_length
        self.dec_max_length = args.dec_max_length

        # Target
        #'Hillary Clinton', 'Legalization of Abortion', 'Atheism',
        #'Climate Change is a Real Concern', 'Feminist Movement'
        if args.target_knowledge_made_by_llm:
            self.llm_target_knowledge = pickle.load(open("data/target_knowledge_made_by_llm.pkl", "rb"))
            self.enc_max_length = 512

        self.wiki_path = args.with_wiki
        if self.wiki_path:
            self.wiki_dict = pickle.load(open("data/wiki_dict(added_sem16t6).pkl", "rb"))
            self.enc_max_length = 512

        self.tokenizer = tokenizer
        self.is_train = is_train
        self.args = args
        self.train_mode = train_mode

        # AGAINST, FAVOR, NONE
        # self.label_mapping_talbe = {
        #     0: 'negative',
        #     1: 'positive',
        #     2: 'neutral'
        # }
        # self.label_ori_to_new = {
        #     'AGAINST': 'negative',
        #     'FAVOR': 'positive',
        #     'NONE' : 'neutral',
        # }
        self.label_mapping_talbe = {
            0: 'negative',
            1: 'positive',
            2: 'neutral'
        }

        self.gen_datas = dict()
        self.preprocess_data()

    def generate_output_template(self, topic, reason, stance, is_train, train_mode, args):
        if not is_train:
            return f"Topic is {topic}. Stance is {stance}."

        offset_begin = None
        offset_end = None

        if train_mode == "stance_prediction":
            output_template = f"Topic is {topic}. Stance is {stance}."
        elif train_mode == "unlikelihood_training":
            # output_template = f"The reasons are as follows. {reason} so, Stance is "
            output_template = f"Topic is {topic}. Stance is "

            offset_begin = len(output_template)
            output_template += f"{stance}"
            offset_end = len(output_template)
            output_template += f"."
        elif train_mode == "topic_unlikelihood_training":
            # output_template = f"The reasons are as follows. {reason} so, Stance is "
            output_template = f"Topic is "
            offset_begin = len(output_template)
            output_template += f"{topic}"
            offset_end = len(output_template)
            output_template += f". Stance is {stance}."
        elif train_mode == "shuffling":
            output_template = f"Topic is {topic}. Stance is {stance}."
        elif train_mode == "topic_prediction":
            output_template = f"Topic is {topic}. Stance is {stance}."
        elif train_mode == "relation_prediction":
            pass
        else:
            raise ValueError(f"There is no {train_mode} in abblation_study.")

        return output_template, (offset_begin, offset_end)

    def preprocess_data(self):
        self.posts = self.data['post'].tolist()
        self.topics = self.data['topic_str'].tolist()
        self.topic_list = self.data['topic_str'].unique().tolist()
        self.labels = self.data['label'].tolist()
        self.labels = [self.label_mapping_talbe[x] for x in self.labels]
        self.new_topics = self.data['new_topic'].tolist()

        self.gen_datas['input_ids'] = list()
        self.gen_datas['decoder_input_ids'] = list()
        self.gen_datas['labels'] = list()
        self.gen_datas['attention_mask'] = list()
        self.gen_datas['negative_loss_mask'] = list()

        self.max_length_in_data = 0

        for i in range(len(self.labels)):
            if not self.is_train:
                input_template = f"Topic is {self.topics[i]}. Stance is <stance>. </s></s>{self.posts[i]}"

                if self.args.target_knowledge_made_by_llm:
                    input_template += f"</s></s>" + self.llm_target_knowledge[self.new_topics[i]]

                if self.wiki_path:
                    input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

                output_template = self.generate_output_template(topic=self.topics[i],
                                                                reason=None,
                                                                stance=self.labels[i],
                                                                is_train=False,
                                                                train_mode=self.train_mode,
                                                                args=self.args)

                input_text_tok = self.tokenizer(input_template, padding="max_length", max_length=self.enc_max_length,
                                                truncation=True)
                output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                 max_length=self.dec_max_length - 800,
                                                 truncation=True, return_offsets_mapping=True)

                self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
                self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
                self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))
                continue

            if self.train_mode == "stance_prediction":
                input_template = f"Topic is {self.topics[i]}. Stance is <stance>. </s></s>{self.posts[i]}"

                if self.args.target_knowledge_made_by_llm:
                    input_template += f"</s></s>" + self.llm_target_knowledge[self.new_topics[i]]

                if self.wiki_path:
                    input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

                output_template, _ = self.generate_output_template(topic=self.topics[i],
                                                                   reason=None,
                                                                   stance=self.labels[i],
                                                                   is_train=self.is_train,
                                                                   train_mode=self.train_mode,
                                                                   args=self.args)

                input_text_tok = self.tokenizer(input_template, padding="max_length", max_length=self.enc_max_length,
                                                truncation=True)
                output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                 max_length=self.dec_max_length - 800,
                                                 truncation=True, return_offsets_mapping=True)

                self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
                self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
                self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))
            elif self.train_mode == "topic_prediction":
                input_template = f"Topic is <topic>. Stance is {self.labels[i]}. </s></s>{self.posts[i]}"

                if self.args.target_knowledge_made_by_llm:
                    input_template += f"</s></s>" + self.llm_target_knowledge[self.new_topics[i]]

                if self.wiki_path:
                    input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

                output_template, _ = self.generate_output_template(topic=self.topics[i],
                                                                   reason=None,
                                                                   stance=self.labels[i],
                                                                   is_train=self.is_train,
                                                                   train_mode=self.train_mode,
                                                                   args=self.args)
                input_text_tok = self.tokenizer(input_template, padding="max_length", max_length=self.enc_max_length,
                                                truncation=True)
                output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                 max_length=self.dec_max_length - 800,
                                                 truncation=True, return_offsets_mapping=True)

                self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
                self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
                self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))
            elif self.train_mode == "unlikelihood_training":
                input_template = f"Topic is {self.topics[i]}. Stance is <stance>. </s></s>{self.posts[i]}"

                if self.args.target_knowledge_made_by_llm:
                    input_template += f"</s></s>" + self.llm_target_knowledge[self.new_topics[i]]

                if self.wiki_path:
                    input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

                fake_labels = ['negative', 'positive', 'neutral']
                fake_labels.remove(self.labels[i])

                for stance in fake_labels:
                    output_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                            reason=None,
                                                                            stance=stance,
                                                                            is_train=self.is_train,
                                                                            train_mode=self.train_mode,
                                                                            args=self.args)

                    input_text_tok = self.tokenizer(input_template, padding="max_length",
                                                    max_length=self.enc_max_length,
                                                    truncation=True)
                    output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                     max_length=self.dec_max_length - 800,
                                                     truncation=True, return_offsets_mapping=True)

                    neg_loss_mask = torch.zeros_like(torch.Tensor(output_text_tok['attention_mask']))

                    for j in range(len(output_text_tok["offset_mapping"])):
                        if (output_text_tok["offset_mapping"][j][0] >= offset[0]) and \
                                (output_text_tok["offset_mapping"][j][1] <= offset[1]):
                            neg_loss_mask[j] = 1.0

                    self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
                    self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
                    self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))
                    self.gen_datas["negative_loss_mask"].append(neg_loss_mask)
            elif self.train_mode == "topic_unlikelihood_training":
                input_template = f"Topic is <topic>. Stance is {self.labels[i]}. </s></s>{self.posts[i]}"

                if self.args.target_knowledge_made_by_llm:
                    input_template += f"</s></s>" + self.llm_target_knowledge[self.new_topics[i]]

                if self.wiki_path:
                    input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

                fake_labels = self.topic_list[:]
                random.shuffle(fake_labels)

                try:
                    fake_labels.remove(self.new_topics[i])
                except Exception:
                    pass

                # for topic in fake_labels[:self.args.epoch]:
                for topic in fake_labels[:self.args.epoch]:
                    output_template, offset = self.generate_output_template(topic=topic,
                                                                            reason=None,
                                                                            stance=self.labels[i],
                                                                            is_train=self.is_train,
                                                                            train_mode=self.train_mode,
                                                                            args=self.args)

                    input_text_tok = self.tokenizer(input_template, padding="max_length",
                                                    max_length=self.enc_max_length,
                                                    truncation=True)
                    output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                     max_length=self.dec_max_length - 800,
                                                     truncation=True, return_offsets_mapping=True)

                    neg_loss_mask = torch.zeros_like(torch.Tensor(output_text_tok['attention_mask']))

                    for j in range(len(output_text_tok["offset_mapping"])):
                        if (output_text_tok["offset_mapping"][j][0] >= offset[0]) and \
                                (output_text_tok["offset_mapping"][j][1] <= offset[1]):
                            neg_loss_mask[j] = 1.0

                    self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
                    self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
                    self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))
                    self.gen_datas["negative_loss_mask"].append(neg_loss_mask)

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        outputs = dict()

        outputs['input_ids'] = self.gen_datas['input_ids'][idx]
        outputs['attention_mask'] = self.gen_datas['attention_mask'][idx]
        outputs['labels'] = self.gen_datas['labels'][idx]
        outputs['stances'] = self.labels[idx]

        if self.train_mode == "unlikelihood_training" or self.train_mode == "topic_unlikelihood_training":
            outputs['negative_loss_mask'] = self.gen_datas['negative_loss_mask'][idx]

        return outputs
class VastLLMDataset(Dataset):
    def __init__(self, data, tokenizer, inference_model_tpye, inference_method, args, train_mode, is_train=True, wiki_path=False):
        super().__init__()

        self.data = data

        self.label_mapping_talbe = {
            0: 'negative',
            1: 'positive',
            2: 'neutral'
        }

        # [ORI, INST]
        self.inference_model_type = inference_model_tpye
        # [zero_shot, few_shot]
        self.inference_method = inference_method

        self.tokenizer = tokenizer
        self.gen_datas = dict()
        self.preprocess_data()

    def generate_output_template(self, topic, reason, stance, is_train, train_mode, args):
        if not is_train:
            return f"Topic is {topic}. Stance is {stance}."

        offset_begin = None
        offset_end = None

        if train_mode == "stance_prediction":
            output_template = f"Topic is {topic}. Stance is {stance}."
        elif train_mode == "unlikelihood_training":
            # output_template = f"The reasons are as follows. {reason} so, Stance is "
            output_template = f"Topic is {topic}. Stance is "

            offset_begin = len(output_template)
            output_template += f"{stance}"
            offset_end = len(output_template)
            output_template += f"."
        elif train_mode == "topic_unlikelihood_training":
            # output_template = f"The reasons are as follows. {reason} so, Stance is "
            output_template = f"Topic is "
            offset_begin = len(output_template)
            output_template += f"{topic}"
            offset_end = len(output_template)
            output_template += f". Stance is {stance}."
        elif train_mode == "shuffling":
            output_template = f"Topic is {topic}. Stance is {stance}."
        elif train_mode == "topic_prediction":
            output_template = f"Topic is {topic}. Stance is {stance}."
        elif train_mode == "relation_prediction":
            pass
        else:
            raise ValueError(f"There is no {train_mode} in abblation_study.")

        return output_template, (offset_begin, offset_end)

    def preprocess_data(self):
        self.posts = self.data['post'].tolist()
        self.topics = self.data['topic_str'].tolist()
        self.topic_list = self.data['topic_str'].unique().tolist()
        self.labels = self.data['label'].tolist()
        # self.labels = [self.label_mapping_talbe[x] for x in self.labels]
        self.new_topics = self.data['new_topic'].tolist()
        self.data_type_marks = self.data['seen?'].tolist()

        # if (self.args.is_long or self.args.is_short) and self.is_train:
        #     self.reasons = self.data['ChatGPT'].tolist()

        self.gen_datas['input_ids'] = list()
        self.gen_datas['decoder_input_ids'] = list()
        self.gen_datas['labels'] = list()
        self.gen_datas['attention_mask'] = list()
        self.gen_datas['data_type_mark'] = torch.LongTensor(self.data_type_marks)
        self.gen_datas['negative_loss_mask'] = list()
        self.gen_datas['templated_sentences'] = list()

        self.max_length_in_data = 0

        def zero_shot_template(topic, post):
            user_template = f"""Predict the expressed stance on a topic from the context of the post and related topics. 
                        Please provide your best objective judgment based on the content of the post and your all background knowledge. The output must be expressed as a single number. 
                        With just one number. If negative: 0, If positive: 1, If neutral: 2
                        Topic: {self.topics[i]}
                        Post: {self.posts[i]}
                        """
            return user_template


        def few_shot_template(topic, post):
            concated_few_shots = ""
            for idx, example in enumerate(few_shots):
                concated_few_shots += f"\nExample {idx+1}:\nTopic: {example['Topic']}\nPost: {example['Post']}\nStance: {example['Stance']}\n"

            user_template = f"""Predict the expressed stance on a topic from the context of the post and related topics. 
                        Please provide your best objective judgment based on the content of the post and your all background knowledge. 
                        The output must be expressed as a single number. With just one number. If negative: 0, If positive: 1, If neutral: 2. 
                        I will give you some examples of sets of posts, topics, and correct stance labels to help you make your decision.
                        
                        First, we will show examples of existing datasets. The total number of examples is 24. And then at the end you will be provided with the data we need to detect the position. Please refer to the following few shot examples :
                        
                        {concated_few_shots}
        
                        And now predict the stance between topic and post from the last following data: 
                        Topic: {self.topics[i]}
                        Post: {self.posts[i]}
                        """
            return user_template

        for i in range(len(self.labels)):
            if self.inference_method == "zero_shot":
                input_template = zero_shot_template(self.topics[i], self.posts[i])
            elif self.inference_method == "few_shot":
                input_template = few_shot_template(self.topics[i], self.posts[i])

            self.gen_datas['templated_sentences'].append(input_template)
            self.gen_datas['labels'].append(self.labels[i])


        if self.inference_model_type == "ORI":
            for i in range(len(self.gen_datas['templated_sentences'])):
                self.gen_datas['templated_sentences'][i] += "stance:"
            output_text_tok = self.tokenizer(self.gen_datas['templated_sentences'], padding=True, return_tensors="pt")
        elif "vicuna" in self.tokenizer.name_or_path:
            temp_lists=[]
            system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "

            for sentence in self.gen_datas['templated_sentences']:
                temp_lists.append(f"{system_prompt}USER: {sentence} ASSISTANT:\nstance:")
            output_text_tok = self.tokenizer(temp_lists, padding=True, return_tensors="pt")
        elif self.inference_model_type == "INST":
            temp_lists=[]

            for sentence in self.gen_datas['templated_sentences']:
                messages = [
                    {"role": "user", "content": sentence},
                ]
                input_data = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                temp_lists.append(input_data+"stance:")

            output_text_tok = self.tokenizer(temp_lists, padding=True, return_tensors="pt")

        self.gen_datas['input_ids'] = torch.LongTensor(output_text_tok.input_ids)
        self.gen_datas['attention_mask'] = torch.LongTensor(output_text_tok.attention_mask)
        self.gen_datas['labels'] = torch.LongTensor(self.gen_datas['labels'])


    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        outputs = dict()

        outputs['input_ids'] = self.gen_datas['input_ids'][idx]
        outputs['attention_mask'] = self.gen_datas['attention_mask'][idx]
        outputs['labels'] = self.gen_datas['labels'][idx]
        outputs['data_type_mark'] = self.gen_datas['data_type_mark'][idx]

        return outputs


class VastReasoningDataset(Dataset):
    def __init__(self, data, tokenizer, args, is_train=True, is_long=True, is_relation_prediction=False,
                 train_mode=None, wiki_path=None):
        """
            train_mode = ['predict_stance', 'predict_topic', 'negative_stance', 'generate_reason']
        """
        super().__init__()

        self.data = data

        self.enc_max_length = args.enc_max_length
        self.dec_max_length = args.dec_max_length

        if args.with_topic_str_chatgpt:
            self.topic_str_chatgpt = pickle.load(open("data/topic_str_chatgpt.pkl", "rb"))
            self.enc_max_length = 400

        if args.with_new_topic:
            self.new_topic_chatgpt = pickle.load(open("data/new_topic_chatgpt.pkl", "rb"))
            self.enc_max_length = 400

        self.wiki_path = args.with_wiki
        if self.wiki_path:
            self.wiki_dict = pickle.load(open("data/wiki_dict.pkl", "rb"))
            self.enc_max_length = 512

        self.tokenizer = tokenizer

        self.is_train = is_train
        self.is_long = is_long
        self.is_relation_prediction = is_relation_prediction
        self.args = args

        self.label_mapping_talbe = {
            0: 'negative',
            1: 'positive',
            2: 'neutral'
        }

        self.gen_datas = dict()
        self.preprocess_data()

    def generate_output_template(self, topic, reason, stance, is_train, train_mode, reasoning_is_long, args,
                                 relation_prediction_label=0, is_relation_prediciton=False):
        offset_begin = None
        offset_end = None
        relation_prediction_label_table = {0: 'consistent', 1: 'inconsistent'}

        if reasoning_is_long:
            output_template = f"Topic is {topic}. Stance is {stance}. <reasoning_long> {reason}"
        elif is_relation_prediciton:
            output_template = f"Topic is {topic}. Stance is {stance}. <relation_prediction> The relationship between Topic, Stance, sentences and Reasoning is {relation_prediction_label_table[relation_prediction_label]}."
        else:
            output_template = f"Topic is {topic}. Stance is {stance}. <reasoning_short> {reason}"

        return output_template, (offset_begin, offset_end)

    def preprocess_data(self):
        self.posts = self.data['post'].tolist()
        self.topics = self.data['topic_str'].tolist()
        self.labels = self.data['label'].tolist()
        self.labels = [self.label_mapping_talbe[x] for x in self.labels]
        self.new_topics = self.data['new_topic'].tolist()
        self.data_type_marks = self.data['seen?'].tolist()

        if self.is_long:
            self.reasons = self.data['ChatGPT_long'].tolist()
        else:
            self.reasons = self.data['ChatGPT_short'].tolist()

        self.gen_datas['input_ids'] = list()
        self.gen_datas['decoder_input_ids'] = list()
        self.gen_datas['labels'] = list()
        self.gen_datas['attention_mask'] = list()
        self.gen_datas['data_type_mark'] = torch.LongTensor(self.data_type_marks)
        self.gen_datas['stance_loss_mask'] = list()

        self.max_length_in_data = 0

        for i in range(len(self.labels)):
            input_template = f"Topic is {self.topics[i]}. Stance is <stance>. </s></s>{self.posts[i]}"

            if self.args.with_topic_str_chatgpt:
                input_template += f"</s></s>" + self.topic_str_chatgpt[self.topics[i]]

            if self.args.with_new_topic:
                input_template += f"</s></s>" + self.new_topic_chatgpt[self.new_topics[i]]

            if self.wiki_path:
                input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

            if self.is_long:
                output_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                        reason=self.reasons[i],
                                                                        stance=self.labels[i],
                                                                        is_train=self.is_train,
                                                                        train_mode="stance_prediction",
                                                                        reasoning_is_long=True,
                                                                        args=self.args)
            elif self.is_relation_prediction:
                if random.random() >= 0.5:
                    coin = random.randint(0, len(self.labels))
                    while coin == i:
                        coin = random.random(0, len(self.labels))

                    input_template += f"</s></s>" + self.reasons[coin]
                    output_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                            reason=self.reasons[i],
                                                                            stance=self.labels[i],
                                                                            is_train=self.is_train,
                                                                            is_relation_prediction=self.is_relation_prediction,
                                                                            train_mode="stance_prediction",
                                                                            relation_prediction_label=1,
                                                                            reasoning_is_long=True,
                                                                            args=self.args)
                else:
                    input_template += f"</s></s>" + self.reasons[i]
                    output_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                            reason=self.reasons[i],
                                                                            stance=self.labels[i],
                                                                            is_train=self.is_train,
                                                                            is_relation_prediction=self.is_relation_prediction,
                                                                            train_mode="stance_prediction",
                                                                            relation_prediction_label=0,
                                                                            reasoning_is_long=True,
                                                                            args=self.args)

            else:
                output_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                        reason=self.reasons[i],
                                                                        stance=self.labels[i],
                                                                        is_train=self.is_train,
                                                                        train_mode="stance_prediction",
                                                                        reasoning_is_long=False,
                                                                        args=self.args)

            input_text_tok = self.tokenizer(input_template, padding="max_length", max_length=self.enc_max_length,
                                            truncation=True)
            if self.is_long:
                output_text_tok = self.tokenizer(output_template, padding="max_length", max_length=self.dec_max_length,
                                                 truncation=True, return_offsets_mapping=True)
            else:
                output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                 max_length=self.dec_max_length - 300,
                                                 truncation=True, return_offsets_mapping=True)

            self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
            self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
            self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        outputs = dict()

        outputs['input_ids'] = self.gen_datas['input_ids'][idx]
        outputs['attention_mask'] = self.gen_datas['attention_mask'][idx]
        outputs['labels'] = self.gen_datas['labels'][idx]
        outputs['data_type_mark'] = self.gen_datas['data_type_mark'][idx]

        return outputs

class Sem16ReasoningDataset(Dataset):
    def __init__(self, data, tokenizer, args, is_train=True, is_long=True, is_relation_prediction=False,
                 train_mode=None, wiki_path=None):
        """
            train_mode = ['predict_stance', 'predict_topic', 'negative_stance', 'generate_reason']
        """
        super().__init__()

        self.data = data

        self.enc_max_length = args.enc_max_length
        self.dec_max_length = args.dec_max_length

        if args.target_knowledge_made_by_llm:
            self.llm_target_knowledge = pickle.load(open("data/target_knowledge_made_by_llm.pkl", "rb"))
            self.enc_max_length = 512

        self.wiki_path = args.with_wiki
        if self.wiki_path:
            self.wiki_dict = pickle.load(open("data/wiki_dict(added_sem16t6).pkl", "rb"))
            self.enc_max_length = 512

        self.tokenizer = tokenizer

        self.is_train = is_train
        self.is_long = is_long
        self.is_relation_prediction = is_relation_prediction
        self.args = args

        self.label_mapping_talbe = {
            0: 'negative',
            1: 'positive',
            2: 'neutral'
        }

        self.gen_datas = dict()
        self.preprocess_data()

    def generate_output_template(self, topic, reason, stance, is_train, train_mode, reasoning_is_long, args,
                                 relation_prediction_label=0, is_relation_prediciton=False):
        offset_begin = None
        offset_end = None
        relation_prediction_label_table = {0: 'consistent', 1: 'inconsistent'}

        if reasoning_is_long:
            output_template = f"Topic is {topic}. Stance is {stance}. <reasoning_long> {reason}"
        elif is_relation_prediciton:
            output_template = f"Topic is {topic}. Stance is {stance}. <relation_prediction> The relationship between Topic, Stance, sentences and Reasoning is {relation_prediction_label_table[relation_prediction_label]}."
        else:
            output_template = f"Topic is {topic}. Stance is {stance}. <reasoning_short> {reason}"

        return output_template, (offset_begin, offset_end)

    def preprocess_data(self):
        self.posts = self.data['post'].tolist()
        self.topics = self.data['topic_str'].tolist()
        self.labels = self.data['label'].tolist()
        self.labels = [self.label_mapping_talbe[x] for x in self.labels]
        self.new_topics = self.data['new_topic'].tolist()
        self.data.fillna(1, inplace=True)

        if self.is_long:
            self.reasons = self.data['ChatGPT_long'].tolist()
        else:
            self.reasons = self.data['ChatGPT_short'].tolist()

        self.gen_datas['input_ids'] = list()
        self.gen_datas['decoder_input_ids'] = list()
        self.gen_datas['labels'] = list()
        self.gen_datas['attention_mask'] = list()
        self.gen_datas['stance_loss_mask'] = list()

        self.max_length_in_data = 0

        for i in range(len(self.labels)):
            input_template = f"Topic is {self.topics[i]}. Stance is <stance>. </s></s>{self.posts[i]}"

            if self.args.target_knowledge_made_by_llm:
                input_template += f"</s></s>" + self.llm_target_knowledge[self.new_topics[i]]

            if self.wiki_path:
                input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

            if self.is_long:
                output_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                        reason=self.reasons[i],
                                                                        stance=self.labels[i],
                                                                        is_train=self.is_train,
                                                                        train_mode="stance_prediction",
                                                                        reasoning_is_long=True,
                                                                        args=self.args)
            elif self.is_relation_prediction:
                if random.random() >= 0.5:
                    coin = random.randint(0, len(self.labels))
                    while coin == i:
                        coin = random.random(0, len(self.labels))

                    input_template += f"</s></s>" + self.reasons[coin]
                    output_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                            reason=self.reasons[i],
                                                                            stance=self.labels[i],
                                                                            is_train=self.is_train,
                                                                            is_relation_prediction=self.is_relation_prediction,
                                                                            train_mode="stance_prediction",
                                                                            relation_prediction_label=1,
                                                                            reasoning_is_long=True,
                                                                            args=self.args)
                else:
                    input_template += f"</s></s>" + self.reasons[i]
                    output_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                            reason=self.reasons[i],
                                                                            stance=self.labels[i],
                                                                            is_train=self.is_train,
                                                                            is_relation_prediction=self.is_relation_prediction,
                                                                            train_mode="stance_prediction",
                                                                            relation_prediction_label=0,
                                                                            reasoning_is_long=True,
                                                                            args=self.args)

            else:
                output_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                        reason=self.reasons[i],
                                                                        stance=self.labels[i],
                                                                        is_train=self.is_train,
                                                                        train_mode="stance_prediction",
                                                                        reasoning_is_long=False,
                                                                        args=self.args)

            input_text_tok = self.tokenizer(input_template, padding="max_length", max_length=self.enc_max_length,
                                            truncation=True)
            if self.is_long:
                output_text_tok = self.tokenizer(output_template, padding="max_length", max_length=self.dec_max_length,
                                                 truncation=True, return_offsets_mapping=True)
            else:
                output_text_tok = self.tokenizer(output_template, padding="max_length",
                                                 max_length=self.dec_max_length - 300,
                                                 truncation=True, return_offsets_mapping=True)

            self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
            self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))
            self.gen_datas['labels'].append(torch.LongTensor(output_text_tok.input_ids))

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        outputs = dict()

        outputs['input_ids'] = self.gen_datas['input_ids'][idx]
        outputs['attention_mask'] = self.gen_datas['attention_mask'][idx]
        outputs['labels'] = self.gen_datas['labels'][idx]

        return outputs


class ReasoningGenerationDataset(Dataset):
    def __init__(self, data, tokenizer, args, is_train=True, is_long=True, is_relation_prediction=False,
                 train_mode=None, wiki_path=None):
        """
            train_mode = ['predict_stance', 'predict_topic', 'negative_stance', 'generate_reason']
        """
        super().__init__()

        self.data = data

        self.enc_max_length = args.enc_max_length
        self.dec_max_length = args.dec_max_length

        if args.with_topic_str_chatgpt:
            self.topic_str_chatgpt = pickle.load(open("data/topic_str_chatgpt.pkl", "rb"))
            self.enc_max_length = 400

        if args.with_new_topic:
            self.new_topic_chatgpt = pickle.load(open("data/new_topic_chatgpt.pkl", "rb"))
            self.enc_max_length = 400

        self.wiki_path = args.with_wiki
        if self.wiki_path:
            self.wiki_dict = pickle.load(open("data/wiki_dict.pkl", "rb"))
            self.enc_max_length = 512

        self.tokenizer = tokenizer

        self.is_train = is_train
        self.is_long = is_long
        self.is_relation_prediction = is_relation_prediction
        self.args = args

        self.label_mapping_talbe = {
            0: 'negative',
            1: 'positive',
            2: 'neutral'
        }

        self.gen_datas = dict()
        self.preprocess_data()

    def generate_output_template(self, topic, stance, is_train, train_mode, reasoning_is_long, args,
                                 relation_prediction_label=0, is_relation_prediciton=False):
        offset_begin = None
        offset_end = None
        relation_prediction_label_table = {0: 'consistent', 1: 'inconsistent'}

        if reasoning_is_long:
            decoder_input_template = f"<s>Topic is {topic}. Stance is {stance}. <reasoning_long> "
        else:
            decoder_input_template = f"<s>Topic is {topic}. Stance is {stance}. <reasoning_short> "

        return decoder_input_template, (offset_begin, offset_end)

    def preprocess_data(self):
        self.posts = self.data['post'].tolist()
        self.topics = self.data['topic_str'].tolist()
        self.labels = self.data['label'].tolist()
        self.labels = [self.label_mapping_talbe[x] for x in self.labels]
        self.new_topics = self.data['new_topic'].tolist()
        self.data_type_marks = self.data['seen?'].tolist()

        self.gen_datas['input_ids'] = list()
        self.gen_datas['decoder_input_ids'] = list()
        self.gen_datas['input_template_filled'] = list()
        self.gen_datas['attention_mask'] = list()
        self.gen_datas['data_type_mark'] = torch.LongTensor(self.data_type_marks)
        self.gen_datas['stance_loss_mask'] = list()

        self.max_length_in_data = 0

        for i in range(len(self.labels)):
            input_template = f"Topic is {self.topics[i]}. Stance is <stance>. </s></s>{self.posts[i]}"
            input_template_filled = f"Topic is {self.topics[i]}. Stance is {self.labels[i]}. </s></s>{self.posts[i]}"

            if self.args.with_topic_str_chatgpt:
                input_template += f"</s></s>" + self.topic_str_chatgpt[self.topics[i]]

            if self.args.with_new_topic:
                input_template += f"</s></s>" + self.new_topic_chatgpt[self.new_topics[i]]

            if self.wiki_path:
                input_template += f"</s></s>" + self.wiki_dict[self.new_topics[i]]

            if self.is_long:
                decoder_input_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                                                 stance=self.labels[i],
                                                                                                 is_train=self.is_train,
                                                                                                 train_mode="stance_prediction",
                                                                                                 reasoning_is_long=True,
                                                                                                 args=self.args)
            else:
                decoder_input_template, offset = self.generate_output_template(topic=self.topics[i],
                                                                                                stance=self.labels[i],
                                                                                                is_train=self.is_train,
                                                                                                train_mode="stance_prediction",
                                                                                                reasoning_is_long=False,
                                                                                                args=self.args)

            input_text_tok = self.tokenizer(input_template, padding="max_length", max_length=self.enc_max_length,
                                            truncation=True)

            input_template_filled_tok = self.tokenizer(input_template_filled, padding="max_length", max_length=self.enc_max_length,
                                            truncation=True)
            if self.is_long:
                decoder_input_text_tok = self.tokenizer(decoder_input_template, padding="max_length", max_length=30, add_special_tokens=False,
                                                        truncation=True, return_offsets_mapping=True)
            else:
                decoder_input_text_tok = self.tokenizer(decoder_input_template, padding="max_length", max_length=30, add_special_tokens=False,
                                                        truncation=True, return_offsets_mapping=True)

            self.gen_datas['input_ids'].append(torch.LongTensor(input_text_tok.input_ids))
            self.gen_datas['input_template_filled'].append(torch.LongTensor(input_template_filled_tok.input_ids))
            self.gen_datas['decoder_input_ids'].append(torch.LongTensor(decoder_input_text_tok.input_ids))
            self.gen_datas['attention_mask'].append(torch.LongTensor(input_text_tok.attention_mask))

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        outputs = dict()

        outputs['input_ids'] = self.gen_datas['input_ids'][idx]
        outputs['input_template_filled'] = self.gen_datas['input_template_filled'][idx]
        outputs['decoder_input_ids'] = self.gen_datas['decoder_input_ids'][idx]
        outputs['attention_mask'] = self.gen_datas['attention_mask'][idx]

        # for key in outputs.keys():
        #     print(f"{key} -> {outputs[key].shape}")

        return outputs