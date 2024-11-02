import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os 
import pandas as pd


# Read the data
s3_path = 's3://hugging-face-multiclass-textclassification-bucket24/training_data/newsCorpora.csv'
df = pd.read_csv(s3_path, sep='\t',names=['ID','TITLE','URL','PUBLISHER','CATEGORY','STORY','HOSTNAME','TIMESTAMP'])


df = df[['TITLE','CATEGORY']]

my_dict = {
    'e': 'Entertainment',
    'b': 'Business',
    't': 'Science',
    'm': 'Health'
}

def update_cat(x):
    return my_dict[x]


df['CATEGORY'] = df['CATEGORY'].apply(lambda x: update_cat(x))
# This is just a tip for production: We take a ver small sample to train and test: Test the model without wasting too much money
#df = df.sample(frac=0.05,random_state=1)

#df = df.reset_index(drop=True)
# This is where the tip ends


# Encoding
encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]


df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))

df = df.reset_index(drop=True)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class NewDataset(Dataset):
    def __init__(self,dataframe,tokenizer,max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self,index):
        title = str(self.data.iloc[index,0])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = 'max_length',
            truncation=True, # To chop off long inputs
            return_token_type_ids=True,
            return_attention_mask=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'ids':torch.tensor(ids,dtype=torch.long),
            'mask':torch.tensor(mask,dtype=torch.long),
            'targets':torch.tensor(self.data.iloc[index,2],dtype=torch.long), # We access column 3 -<
        }
    
    def __len__(self):
        return self.len
    
    

train_size = 0.8 # 80% of the data for training
train_dataset = df.sample(frac=train_size,random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

train_dataset.reset_index(drop=True)

print("Full dataset: {}".format(df.shape))
print("Train dataset: {}".format(train_dataset.shape))
print("Test dataset: {}".format(test_dataset.shape))


MAX_LEN = 512
TRAIN_BATCH_SIZE=4
VALID_BATCH_SIZE=2



training_set = NewDataset(train_dataset,tokenizer,MAX_LEN)
testing_set = NewDataset(test_dataset,tokenizer,MAX_LEN)

train_parameters = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers':0,
}

test_parameters = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': True,
    'num_workers':0,
}

training_loader = DataLoader(training_set,**train_parameters)
testing_loader = DataLoader(testing_set,**test_parameters)


class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super().__init__() # Calling the init function of the parent class -- torch.nn.Module
        
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased') # Loads a pre-trained distilbert model uncased -> Don't differentiate between upper and lowercase
        
        self.pre_classifier = torch.nn.Linear(768,768) # Add new weights to the model
        
        self.dropout = torch.nn.Dropout(0.3) # Regularization technique preventing overfitting
        
        self.classifier = torch.nn.Linear(768,4) # Takes 768 inputs and outputs 4 - Entertainment, Business, Science of Health
        
        
        
    def forward(self,input_ids,attention_mask):
        
        output_1 = self.l1(input_ids = input_ids,attention_mask = attention_mask)
        
        hidden_state = output_1[0]
        
        pooler = hidden_state[:,0] # Hidden state associated with cls token
        
        pooler = self.pre_classifier(pooler)
        
        pooler = torch.nn.ReLU()(pooler) # Activation function --> Helps the model solve more complex relationships
        
        pooler = self.dropout(pooler) # Previous layer
        
        output = self.classifier(pooler)
        
        return output
    
    
    
def calculate_accu(big_index,targets):
    n_correct = (big_index==targets).sum().item()
    return n_correct


def train(epoch,model,device,training_loader,optimizer,loss_function):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    
    for _,data in enumerate(training_loader,0):
        ids = data['ids'].to(device,dtype = torch.long)
        mask = data['mask'].to(device,dtype = torch.long)
        targets = data['targets'].to(device,dtype = torch.long)
        
        outputs = model(ids,mask) # Calling the forward function
        
        loss = loss_function(outputs,targets)
        tr_loss += loss.item()
        big_val,big_idx = torch.max(outputs.data,dim=1)
        n_correct += calculate_accu(big_idx,targets)
        
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)
        
        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Training loss per 5000 steps: {loss_step}")
            print(f"Training accuracy per 5000 steps: {accu_step}")
            
            
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step() # Adjust weight according to opt. algorythm
        
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Acc Epoch: {epoch_accu}")
    
    return


def valid(epoch,model,testing_loader,device,loss_function):
    
    model.eval()
    
    n_correct = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    
    with torch.no_grad():
        for _,data in enumerate(testing_loader,0):
            ids = data['ids'].to(device,dtype = torch.long)
            mask = data['mask'].to(device,dtype = torch.long)
            targets = data['targets'].to(device,dtype = torch.long)

            outputs = model(ids,mask).squeeze()

            loss = loss_function(outputs,targets)
            tr_loss += loss.item()
            big_val,big_idx = torch.max(outputs.data,dim=1)
            n_correct += calculate_accu(big_idx,targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 1000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Training loss per 5000 steps: {loss_step}")
                print(f"Training accuracy per 5000 steps: {accu_step}")

            
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Acc Epoch: {epoch_accu}")
    return
            
        
        
def main():
    print("Start")
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs",type=int,default=10) # Passing paramenters in Sagemaker
    parser.add_argument("--train_batch_size",type=int,default=4) # Passing paramenters in Sagemaker
    parser.add_argument("--valid_batch_size",type=int,default=2) # Passing paramenters in Sagemaker
    parser.add_argument("--learning_rate",type=float,default=5e-5) # Passing paramenters in Sagemaker
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Checks if you have a GPU (cuda), otherwise sets device to cpu
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    model = DistilBERTClass()
    
    model.to(device)
    
    optimizer = torch.optim.Adam(params = model.parameters(),lr=args.learning_rate)
    
    loss_function = torch.nn.CrossEntropyLoss()
    
    # Train loop
    for epoch in range(args.epochs):
        print(f"starting epoch: {epoch}")
        
        train(epoch,model,device,training_loader,optimizer,loss_function)
        
        valid(epoch,model,testing_loader,device,loss_function)
    
    output_dir = os.environ['SM_MODEL_DIR'] # SageMaker output directory
    
    output_model_file = os.path.join(output_dir,'pytorch_distilbert_news.bin')
    
    output_vocab_file = os.path.join(output_dir,'vocab_distilbert_news.bin')
    
    torch.save(model.state_dict(),output_model_file)
    
    tokenizer.save_vocabulary(output_vocab_file)
    
    
if __name__ == '__main__':
    main()