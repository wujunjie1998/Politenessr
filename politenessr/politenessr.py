import pandas as pd
import numpy as np
import torch
from torch.utils.data import (DataLoader, TensorDataset)
import os
from pytorch_pretrained_bert import BertTokenizer
from pytorch_transformers import BertForSequenceClassification
from multiprocessing import Pool, cpu_count
from .convert_examples_to_features import convert_example_to_feature
from tqdm import tqdm
from .regression_processor import RegressionProcessor
from .download_features import fetch_pretrained_model
import logging
from os.path import expanduser
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class Politenessr:
    def __init__(self,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 is_paralleled=False, BATCH_SIZE = 128, CPU_COUNT=1, CHUNKSIZE=1):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        model_path = os.path.join(os.path.dirname(__file__),"politeness_model.bin")
        self.model_type = 'politenessr'

        if not os.path.isfile(model_path):
            logger.info(f'Model {self.model_type} does not exist at {model_path}. Try to download it now.')
            model = 'politeness_model'
            fetch_pretrained_model(model, model_path)

        if self.device.type == "cpu":
            model_state_dict = torch.load(model_path, map_location=self.device.type)
        else:
            model_state_dict = torch.load(model_path)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-cased', state_dict=model_state_dict, num_labels=1)
        if is_paralleled:
            if self.device.type == "cpu":
                print("Data parallel is not available with cpus")
            else:
                self.model = torch.nn.DataParallel(self.model)

        self.model.to(device)
        self.model.eval()
        self.batch_size = BATCH_SIZE
        self.cpu_count = CPU_COUNT
        self.chunksize = CHUNKSIZE

    def predict(self,text):
        text1 = pd.DataFrame(text, columns=['body'])
        train_df_bert = pd.DataFrame({
            'id': range(len(text)),
            'label': 0,
            'alpha': ['a'] * text1.shape[0],
            'text': text1['body'].replace(r'\n', ' ', regex=True)
        })
        train_df_bert.to_csv('train_regression.tsv', sep='\t', index=False, header=False)

        processor = RegressionProcessor()
        train_examples = processor.get_train_examples('')
        train_examples_len = len(train_examples)
        os.remove('train_regression.tsv')

        OUTPUT_MODE = 'regression'
        MAX_SEQ_LENGTH = 128

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        label_map = {'0.0': 0}
        train_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in
                                         train_examples]

        process_count = self.cpu_count
        chunksize = self.chunksize
        #global train_features


        with Pool(process_count) as p:
            train_features = list(tqdm(
                p.imap(convert_example_to_feature, train_examples_for_processing, chunksize),
                total=train_examples_len))

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        bert_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        bert_dataloader = DataLoader(bert_data, batch_size=self.batch_size)
        preds = []

        if self.device.type == 'cpu':
            for step, batch in enumerate(tqdm(bert_dataloader)):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.no_grad():
                    logits = self.model(input_ids, segment_ids, input_mask, labels=None)
                # logits = logits.detach().cpu()

                preds.append(logits)
            preds = [item for sublist in preds for item in sublist]
            preds = np.squeeze(np.array(preds[0]))
        else:
            for step, batch in enumerate(tqdm(bert_dataloader)):
                batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.no_grad():
                    logits = self.model(input_ids, segment_ids, input_mask, labels=None)

                logits = logits[0]
                logits = logits.detach().cpu()
                if len(preds) == 0:
                    preds.append(logits)
                else:
                    preds[0] = np.append(
                        preds[0], logits, axis=0)
            preds = [item for sublist in preds for item in sublist]
            preds = np.squeeze(np.array(preds))
        return preds






















