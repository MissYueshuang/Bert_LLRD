from torch.nn.modules.loss import CrossEntropyLoss
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn 

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score,accuracy_score
from sklearn.utils import class_weight

from nltk.stem import WordNetLemmatizer
import nltk

import warnings
warnings.filterwarnings("ignore")


# do for train and validation data seperately. Take train data for example
def Load_data(X, y, max_length=128, batch_size=3):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    encoded_data_train = tokenizer.batch_encode_plus(
        X, 
        add_special_tokens=True, # the sequences will be encoded with the special tokens relative to their model.
        return_attention_mask=True,  
        pad_to_max_length=True, # pad all the titles to certain maximum length.
        max_length=max_length, 
        return_tensors='pt' # return PyTorch.
    ) # it would return a pytorch dataset containing input_ids, attention_mask and token_type_ids

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(y,dtype=torch.long)

    # combine data together (X,y)
    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataloader_train = DataLoader(dataset_train, 
                                sampler=RandomSampler(dataset_train), 
                                batch_size=batch_size)  

    return dataloader_train

# performance matrix
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def acc_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat)

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


def evaluate(dataloader_val):    

    model.eval() # call it first when evaluate the model
    
    loss_val_total = 0
    predictions = []
    true_vals = []

    for batch in dataloader_val:        
        batch = tuple(b.to(device) for b in batch) # first to gpu        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2], # true y
                 } # get inputs
        
        # 评估的时候不需要更新参数、计算梯度
        with torch.no_grad():     
            outputs = model(**inputs) # feed into model
            
        # loss = outputs[0] # first return is loss
        logits = outputs[1] # second return is logistic propability p
        labels = inputs['labels']

        # loss_fct = FocalLoss(args.out_features,gamma=args.gamma,size_average=False)       
        # loss = loss_fct(logits.view(-1, args.out_features), labels.view(-1))        
        # loss = outputs[0]
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        loss = loss_fct(logits.view(-1, args.out_features), labels.view(-1))
        # loss = outputs[0]

        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = labels.detach().cpu().numpy()

        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

def AdamW_LLRD(model):
    
   opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
   named_parameters = list(model.named_parameters())

   # According to AAAMLP book by A. Thakur, we generally do not use any decay

   # for bias and LayerNorm.weight layers.
   no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
   init_lr = 3.5e-6
   head_lr = 3.6e-6 # pooler
   lr = init_lr

   # === Pooler and regressor ======================================================  
   params_0 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n)
               and any(nd in n for nd in no_decay)]
   params_1 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n)
               and not any(nd in n for nd in no_decay)]

   head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
   opt_parameters.append(head_params)

   head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}    
   opt_parameters.append(head_params)

   # === 12 Hidden layers ==========================================================
   for layer in range(11,-1,-1):        
       params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n
                   and any(nd in n for nd in no_decay)]
       params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n
                   and not any(nd in n for nd in no_decay)]

       layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
       opt_parameters.append(layer_params)  

       layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
       opt_parameters.append(layer_params)  

       lr *= 0.9    

   # === Embeddings layer ==========================================================
   params_0 = [p for n,p in named_parameters if "embeddings" in n
               and any(nd in n for nd in no_decay)]
   params_1 = [p for n,p in named_parameters if "embeddings" in n
               and not any(nd in n for nd in no_decay)]

   embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
   opt_parameters.append(embed_params)

   embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
   opt_parameters.append(embed_params)        

   return AdamW(opt_parameters, lr=init_lr)

class BertClassifier(nn.Module):
    
  def __init__(self, n_classes, dropout=0.1):

    super(BertClassifier, self).__init__()
    self.bert = BertModel.from_pretrained("bert-base-uncased")
    self.drop = nn.Dropout(p=dropout)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    returned = self.bert(        
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    pooled_output = returned["pooled_output"]
    output = self.drop(pooled_output)
    
    return self.out(output)


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    
    def __init__(self, class_num, alpha=None, gamma=5, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        log_p = (P * class_mask).sum(1).view(-1,1)
        probs = log_p.exp()
        batch_loss = -alpha * (torch.pow((1-probs), self.gamma)) * log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def text_preprocessing(tokenized_text):
    try:
        tokenized_text = tokenized_text.split(' ')
    except:
        pass
    stopwords = nltk.corpus.stopwords.words('english')
    tokenized_text = [word.strip().lower() for word in tokenized_text] # remove space and change to lowercase
    tokenized_text = [word for word in tokenized_text if word not in stopwords and len(word)>1] # remove stopwords
    wordnet_lemmatizer = WordNetLemmatizer()
    tokenized_text = [wordnet_lemmatizer.lemmatize(word) for word in tokenized_text] # do lemmatization    
    
    return (' ').join(tokenized_text)


if __name__ == '__main__':  

    parser = argparse.ArgumentParser()   

    ## Required parameters
    parser.add_argument("--parent_path",
                        default=r"D:\LYS\temp files\BERT\output_LLRD",
                        type=str)
    parser.add_argument("--lr",
                        default=1e-5,
                        type=float,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--batch_size",
                        default=4,
                        type=int)    
    parser.add_argument("--num_train_epochs",
                        default=4,
                        type=int)
    parser.add_argument("--num_warmup_steps",
                        default=50,
                        type=int)
    parser.add_argument("--max_length",
                        default=64,
                        type=int)
    parser.add_argument("--out_features",
                        default=5,
                        type=int)
    parser.add_argument("--dense_dropout",
                        default=0.1,
                        type=float)
    parser.add_argument("--do_freeze",
                        default=True,
                        type=bool)
    parser.add_argument("--cut_grad",
                        default=1,
                        type=int)
    parser.add_argument("--gamma",
                        default=2,
                        type=int)
    parser.add_argument("--random_state",
                        default=42,
                        type=int)
    parser.add_argument("--test_size",
                        default=0.1,
                        type=float)
    parser.add_argument("--do_save_model",
                        default=False,
                        type=bool)

    args = parser.parse_args()

    if not os.path.isdir(args.parent_path):
        os.mkdir(args.parent_path)
    os.chdir(args.parent_path)

    with open('config.json','w') as fp:
        json.dump(vars(args),fp, indent=4)


    # prepare data
    # data = pd.read_csv(r'D:\LYS\python_learning\NLP\DBS\data\dataset_translated.csv')
    # data.tags = data.tags.apply(lambda x: " ".join(x.split("  ")))
    # data.drop_duplicates(subset=['tags'],inplace=True)
    # data.tags = data.tags.apply(lambda x:text_preprocessing(x))
    # lyrics = data.tags.values

    # encoder = LabelEncoder()
    # genre = encoder.fit_transform(data.genre.values)
    # label_dict = dict(zip(encoder.classes_,range(len(encoder.classes_))))

    # X_train, X_test, y_train, y_test = train_test_split(lyrics,genre,test_size=args.test_size,random_state=args.random_state)
    data_dir=r'D:\LYS\temp files\BERT'
    train = pd.read_csv(os.path.join(data_dir,'train_NLI_M.csv'),header=None, sep="\t")
    X_train = train.iloc[:,1]
    y_train = train.iloc[:,-1]
    y_train += 2

    test = pd.read_csv(os.path.join(data_dir,'dev_NLI_M.csv'),header=None, sep="\t")
    X_test = test.iloc[:,1]
    y_test = test.iloc[:,-1]
    y_test += 2

    class_weights = list(class_weight.compute_class_weight('balanced',
                                                np.unique(y_train),
                                                y_train))
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    # class_weights = torch.FloatTensor(class_weights).cuda()

    ## to loader
    dataloader_train = Load_data(X_train, y_train, max_length=args.max_length, batch_size=args.batch_size)
    dataloader_validation = Load_data(X_test, y_test, max_length=args.max_length, batch_size=args.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    ## load model
    # model = BertClassifier(args.out_features,dropout=0.2)
    
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", # layer=12
                                            num_labels=args.out_features, # how many class for your y 
                                            output_attentions=False,
                                            output_hidden_states=False)
    model.dropout = nn.Dropout(p=args.dense_dropout)
    model.to(device)
    # config = BertConfig.from_pretrained('bert-base-uncased')
    # config.num_labels = args.out_features
    # model = BertForSequenceClassification(config) 
    # model = model.to(device)

    # if we choose to freeze 
    # freeze all the layers in model.bert, excluding the classifier. 
    for param in model.bert.parameters(): 
        param.requires_grad = args.do_freeze
    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param)
    named_parameters = list(model.named_parameters())
    
    no_decay = ['bias', 'LayerNorm.weight']
    # init_lr = 3.5e-5
    # head_lr = 3.6e-5
    # lr = init_lr

    # old version
    optimizer_parameters = [
        # n is param name and p is prob
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
        # 'lr':head_lr,
        'weight_decay_rate': 0.0},

        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
        # 'lr':head_lr,
        'weight_decay_rate': 0.01}

        ]

    # torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # use AdamW as Optimizer
    # optimizer = AdamW(optimizer_parameters, # model.parameters()
    #                 lr=args.lr, 
    #                 eps=1e-8)
    #                 # weight_decay=0.01)
    
    ## update1.0: using layer-wise lr
    optimizer = AdamW_LLRD(model)

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=len(dataloader_train)*args.num_train_epochs)

    epoch = 0
    seed_log = open("seed_log.txt", 'wt')
    seed_log.write("================\n")
    lrs = []
    curr_best_model = None
    curr_best_dev_accu = 0

    # train all data epoch times
    for _ in trange(args.num_train_epochs, desc="Epoch"):
        epoch += 1
        print("============={}=============".format(epoch))
        model.train()        
        loss_train_total = 0
        for step, batch in enumerate(dataloader_train):             
            
            batch = tuple(b.to(device) for b in batch)            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }       

            outputs = model(**inputs)
            
            logits = outputs[1]
            
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
            # loss_fct = FocalLoss(args.out_features,gamma=args.gamma,size_average=False)
            labels = inputs['labels']
            loss = loss_fct(logits.view(-1, args.out_features), labels.view(-1).to(device))
            # loss = outputs[0]
            loss_train_total += loss.item()

            loss.backward() # backpropagation, calculate gradient


            torch.nn.utils.clip_grad_norm_(model.parameters(), args.cut_grad) # if gradient is less than 1, set gradient to 1 norm

            optimizer.step() # update parameters
            scheduler.step() # update learning rate
            lrs.append(optimizer.param_groups[0]["lr"])

            model.zero_grad() # set gradient to 0
            
        tqdm.write(f'\nEpoch {epoch}')
        seed_log.write("Epoch: {}\n".format(epoch))
        
        loss_train_avg = loss_train_total/len(dataloader_train)    

        train_loss, predictions_train, true_train = evaluate(dataloader_train)
        train_f1 = f1_score_func(predictions_train, true_train)
        train_acc = acc_score_func(predictions_train, true_train)

        tqdm.write(f'Training loss: {loss_train_avg}')
        tqdm.write(f'Training Accuracy Score: {train_acc}')

        seed_log.write("Training loss: {}\n".format(loss_train_avg))
        seed_log.write("Training F1 Score (Weighted): {}\n".format(train_f1))
        seed_log.write("Train Accuracy Score: {}\n".format(train_acc))

        val_loss, predictions_val, true_vals = evaluate(dataloader_validation)
        val_f1 = f1_score_func(predictions_val, true_vals)
        val_acc = acc_score_func(predictions_val, true_vals)

        tqdm.write(f'Validation loss: {val_loss}')
        # tqdm.write(f'F1 Score (Weighted): {val_f1}')
        tqdm.write(f'Validation Accuracy Score: {val_acc}')
        
        seed_log.write("\n")
        seed_log.write("Validation loss: {}\n".format(val_loss))
        seed_log.write("Validation F1 Score (Weighted): {}\n".format(val_f1))
        seed_log.write("Validation Accuracy Score: {}\n".format(val_acc))
        seed_log.write("\n")

        if val_acc > curr_best_dev_accu:
            curr_best_model = model.state_dict()
            curr_best_dev_accu = val_acc
            print("UPDATE MODEL")
        
    
    seed_log.write("=" * 20 + "\n")
    seed_log.close()

    # plot learning rate
    # plt.plot(lrs)
    # # plt.show()
    # plt.savefig('lr.png')

    if args.do_save_model:
        torch.save(curr_best_model.state_dict(), 'finetuned_BERT.model')
