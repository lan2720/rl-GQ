import torch
from torch.utils.data import Dataset
import pickle


class SQuADDataset(Dataset):
    """
    SQuAD dataset

    """
    def __init__(self, data_file):
        self.data = self._load_data(data_file)

    def __len__(self):
        return len(self.data['context'])

    def __getitem__(self, idx):
        #context = torch.from_numpy(self.data['context'][idx]).long()
        sentence = torch.from_numpy(self.data['sentence'][idx]).long()
        answer = torch.from_numpy(self.data['answer'][idx]).long()
        question = torch.from_numpy(self.data['question'][idx]).long()
        lengths = torch.from_numpy(self.data['lengths'][idx]).long()
        ans_start_in_sent = torch.tensor(self.data['ans_start_in_sent'], dtype=torch.long) 
        ans_end_in_sent = torch.tensor(self.data['ans_end_in_sent'], dtype=torch.long) 
        
        sample = {'sentence':sentence, 'answer':answer, 'question':question, 
                  'lengths':lengths, 'ans_start_in_sent':ans_start_in_sent, 'ans_end_in_sent':ans_end_in_sent}

        return sample

    def _load_data(self, data_file):
        data = pickle.load(open(data_file, 'rb'))
        return {'context':data['context'],
                'sentence':data['sentence'],
                'answer':data['answer'],
                'question':data['question'],
                'lengths':data['len'],
                'ans_start_in_sent':data['ans_start_in_sent'],
                'ans_end_in_sent':data['ans_end_in_sent']}


class ToyDataset(Dataset):
    def __init__(self):
        self.data = [[3,5,6,76,4,43,6], [3,5,6], [6,7,4,4,7]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
