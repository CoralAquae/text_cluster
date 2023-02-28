from data_use import *
from model import *
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, TensorDataset


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # read data
    f = open('D:\\学校\\实习\\toutiao_cat_data.txt', 'r', encoding='utf-8')
    text = get_examples(f)
    # 划分训练集测试集
    x_train, x_test = train_test_split(text[:], test_size=0.1, stratify=text['index'])

    # import model
    tokenizer = BertTokenizer.from_pretrained('D:\\学校\\学年论文\\1.16\\chinese_roberta_wwm_ext_pytorch')
    model = BertForSequenceClassification.from_pretrained('D:\\学校\\学年论文\\1.16\\chinese_roberta_wwm_ext_pytorch', num_labels=16).to(device)
    dropout = nn.Dropout(0.05, inplace=False)
    model.dropout = dropout
    print(model)
    for i, param in enumerate(model.parameters()):
        if i < 15:  # 参数冻结
            param.requires_grad = False

    train_encoding = tokenizer(np.array(x_train[['text']].T).tolist()[0], truncation=True, padding=True, max_length=64)
    test_encoding = tokenizer(np.array(x_test[['text']].T).tolist()[0], truncation=True, padding=True, max_length=64)

    train_embedding = NewsDataset(train_encoding,label_change(np.array(x_train[['index']].T).tolist()[0]))
    test_embedding = NewsDataset(test_encoding, label_change(np.array(x_test[['index']].T).tolist()[0]))

    train_loader = DataLoader(train_embedding, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_embedding, batch_size=32, shuffle=True)

    # 优化方法
    # optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-6)
    optim = AdamW(model.parameters(), lr=3e-5, eps = 1e-8)
    # total_steps = len(train_loader) * 1
    # scheduler = get_linear_schedule_with_warmup(optim,
    #                                            num_warmup_steps=0,
    #                                            num_training_steps=total_steps)

    for epoch in range(8):
        print("------------Epoch: %d ----------------" % epoch)
        train(model, train_loader, optim, 0,epoch)
        validation(model, test_dataloader)
        ## 保存模型
        torch.save(model.state_dict(), './model'+str(epoch)+'.pt')

        ## 读取模型
        # state_dict = torch.load('model_name.pth')
        # model.load_state_dict(state_dict['model'])


