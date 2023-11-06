import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForTokenClassification
from sklearn.metrics import euclidean_distances
import torch.nn.functional as F

class CLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']
        self.pad_value = config['pad_value']
        self.mask_value = config['mask_value']
        self.f_context_encoder = AutoModel.from_pretrained(config['bert_path'], 
                                                       local_files_only=False)
        self.gru1 = nn.GRU(input_size=768, hidden_size=768, batch_first=True) 
        
        self.f_context_encoder_hinglish = AutoModelForTokenClassification.from_pretrained("l3cube-pune/hing-bert-lid")


        num_embeddings, self.dim = self.f_context_encoder.embeddings.word_embeddings.weight.data.shape
        self.f_context_encoder.resize_token_embeddings(num_embeddings + 256)

        # self.dim = 768

        self.predictor = nn.Sequential(
            nn.Linear(self.dim, self.num_classes)
        )
        self.g = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            )
    def device(self):
        return self.f_context_encoder.device

    def gen_f_reps(self, sentences):
        '''
        generate vector representations for each turn of conversation
        '''
        batch_size, max_len = sentences.shape[0], sentences.shape[-1]
        sentences = sentences.reshape(-1, max_len)
        mask = 1 - (sentences == (self.pad_value)).long()
        utterance_encoded = self.f_context_encoder(
            input_ids=sentences,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        mask_pos = (sentences == (self.mask_value)).long().max(1)[1]
        mask_outputs = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos, :]
        # feature = torch.dropout(mask_outputs, 0.1, train=self.training)
        feature = mask_outputs
        if self.config['output_mlp']:
            feature = self.g(feature)
        return feature

    def gen_f_reps_ours(self, sentences):
        #sentence is atensor of tokens
        embeddings= self.f_context_encoder_hinglish.bert(sentences).last_hidden_state
        
        embeddings = embeddings.squeeze(dim =0)
        # sentence_embedding = torch.mean(embeddings, dim=1)
        _,sentence_embedding = self.gru1(embeddings)
        
        return sentence_embedding

    def forward(self, reps, centers, score_func):

        num_classes, num_centers = centers.shape[0], centers.shape[1]
        reps = reps.unsqueeze(1).expand(reps.shape[0], num_centers, -1)
        reps = reps.unsqueeze(1).expand(reps.shape[0], num_classes ,num_centers, -1)

        centers = centers.unsqueeze(0).expand(reps.shape[0], -1, -1, -1)
        # batch * turn, num_classes, num_centers
        sim_matrix = score_func(reps, centers)

        # batch * turn, num_calsses
        scores = sim_matrix
        return scores

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.emb_name = 'word_embeddings.weight'

    def attack(self, epsilon=1.):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    def save_checkpoint(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
    def load_checkpoint(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
    
    