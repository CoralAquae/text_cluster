import torch
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

class Classifier_model(nn.Module):
    def __init__(self):
        super(Classifier_model, self).__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device('cpu')
        self.model = BertForSequenceClassification.from_pretrained('hfl/chinese-roberta-wwm-ext-large',
                                                              num_labels=16)
        self.model.to(device)
        dropout = nn.Dropout(0.4, inplace=False)
        self.model.dropout = dropout

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, attention_mask, labels):
        outputs = self.model(x, attention_mask, labels)
        return outputs


def get_encode(text, device, tokenizer, model):
    """
    description: 预训练编码中文文本
    :param text: 要进行编码的文本
    :return: 编码后的文本张量表示
    """
    # 首先使用字符映射器对每个汉字进行映射
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)  # Batch size 1
    outputs = model(input_ids)
    # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    sequence_output = outputs[0]
    return(sequence_output)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train(model, train_loader, optim, scheduler, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    model.train()
    total_train_accuracy = 0
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        # 正向传播
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_train_accuracy += flat_accuracy(logits, label_ids)
        avg_train_accuracy = total_train_accuracy / len(train_loader)

        model.zero_grad()
        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 参数更新
        optim.step()
        # scheduler.step()

        iter_num += 1
        if (iter_num % 100 == 0):
            print("epoch: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    print("Train Accuracy: %.4f" % (avg_train_accuracy))
    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))

def validation(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            logits = outputs[1]

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Test Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_dataloader)))
    print("-------------------------------")


