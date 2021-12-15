from utils import LoadData, SplitData, SCS_dataset, create_data_loader, ProblemClassifier, \
train_epoch, eval_model, test_model
import argparse
import torch
from torch import nn, optim
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
import pandas as pd

DATA_PATH = './datasets/'
MODEL_PATH = './models/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LEN = 64
BATCH_SIZE = 16
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
def main(params):
    if params.train:
        data = LoadData(DATA_PATH+params.train_data)
        df_train, df_valid = SplitData(data)
        train_dataloader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
        valid_dataloader = create_data_loader(df_valid, tokenizer, MAX_LEN, BATCH_SIZE)
        model = ProblemClassifier(2)
        model = model.to(device)

        loss_fn = nn.NLLLoss().to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(train_dataloader)*params.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(params.epochs):

            print(f'Epoch {epoch + 1}/{params.epochs}')
            print('-' * 10)

            train_acc, train_loss = train_epoch(
                model,
                train_dataloader,    
                loss_fn, 
                optimizer, 
                device, 
                scheduler, 
                len(df_train)*2
                )

            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = eval_model(
                model,
                valid_dataloader,
                loss_fn, 
                device, 
                len(df_valid)*2
                )

            print(f'Val   loss {val_loss} accuracy {val_acc}')
            print()
        
        torch.save(model.state_dict(), MODEL_PATH+params.model)
    elif params.test:
        df_test = pd.read_csv(DATA_PATH+params.test_data)
        model = ProblemClassifier(2)
        model.load_state_dict(torch.load(MODEL_PATH+params.model))
        model = model.to(device)
        print(test_model(model, df_test, tokenizer, 64))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT text completion solver')
    parser.add_argument('--train', action='store_const', const=True, default=False)
    parser.add_argument('--test', action='store_const', const=True, default=False)
    parser.add_argument('--test_data', type=str, default='scs_testing.csv')
    parser.add_argument('--train_data', type=str, default='scs_training.csv')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--model', type=str, default='model.torch')

    main(parser.parse_args())