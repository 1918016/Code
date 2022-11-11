import numpy as np
import pandas as pd
import random
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# 全局参数，随机种子，图像尺寸
seed = 114514
np.random.seed(seed)
random.seed(seed)
BATCH_SIZE = 512

hidden_dim = 16
epochs = 1
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print("cuda:0")

df = pd.read_csv("D:/大三上/机器学习/课程设计/Book _Recommendations/train_dataset1.csv")
print('共{}个用户，{}本图书，{}条记录'.format(max(df['user_id']) + 1, max(df['item_id']) + 1, len(df)))
df.head()

class Goodbooks(Dataset):
    def __init__(self, df, mode='training', negs=99):
        super().__init__()

        self.df = df
        self.mode = mode

        self.book_nums = max(df['item_id']) + 1
        self.user_nums = max(df['user_id']) + 1

        self._init_dataset()

    def _init_dataset(self):
        self.Xs = []

        # 下面的两个for循环建立了每一位用户与该用户看过的书籍之间的映射关系，即 :{用户1：用户1看过的书籍, 用户2：用户2看过的书籍, ...}
        self.user_book_map = {}
        for i in range(self.user_nums):
            self.user_book_map[i] = []
        for index, row in self.df.iterrows():
            user_id, book_id = row
            self.user_book_map[user_id].append(book_id)
            # self.user_book_map={用户1：用户1看过的书籍, 用户2：用户2看过的书籍, ...}

        # 对于每一个用户的交互数据，训练集使用除了最后一个item之外的所有item(书籍)，而验证集只使用最后一个item(书籍)
        # 训练集样本结构：(用户id，书籍id，label)
        # label表示是否阅读，是：1，否：0
        if self.mode == 'training':
            for user, items in tqdm.tqdm(self.user_book_map.items()):
                for item in items[:-1]:
                    # 构建正样本，对应label为1
                    self.Xs.append((user, item, 1))
                    # 构建负样本，对应label为0
                    # 正负样本比例为1:3，模拟真实情况下，用户已经阅读过的书籍数小于书籍总数
                    for _ in range(3):
                        while True:
                            neg_sample = random.randint(0, self.book_nums - 1)
                            if neg_sample not in self.user_book_map[user]:
                                self.Xs.append((user, neg_sample, 0))
                                break
        # 验证集样本结构：(用户id，已阅读书籍id，未阅读书籍id)
        elif self.mode == 'validation':
            for user, items in tqdm.tqdm(self.user_book_map.items()):
                if len(items) == 0:
                    continue
                self.Xs.append((user, items[-1]))

    def __getitem__(self, index):
        if self.mode == 'training':
            user_id, book_id, label = self.Xs[index]
            return user_id, book_id, label
        elif self.mode == 'validation':
            user_id, book_id = self.Xs[index]
            # 在所有的当前用户没有看过的书籍中随机抽取99本，之后大概会对这99本排序，对当前用户进行推荐？
            negs = list(random.sample(
                list(set(range(self.book_nums)) - set(self.user_book_map[user_id])),
                k=99
            ))
            return user_id, book_id, torch.LongTensor(negs)

    def __len__(self):
        return len(self.Xs)


# 建立训练和验证dataloader
traindataset = Goodbooks(df, 'training')
validdataset = Goodbooks(df, 'validation')

trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
validloader = DataLoader(validdataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)



# 构建模型
class NCFModel(torch.nn.Module):
    def __init__(self, hidden_dim, user_num, item_num, mlp_layer_num=4, weight_decay=1e-5, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.user_num = user_num
        self.item_num = item_num
        self.mlp_layer_num = mlp_layer_num
        self.weight_decay = weight_decay
        self.dropout = dropout

        # MLP的Embedding层
        self.mlp_user_embedding = torch.nn.Embedding(user_num, hidden_dim * (2 ** (self.mlp_layer_num - 1)))
        self.mlp_item_embedding = torch.nn.Embedding(item_num, hidden_dim * (2 ** (self.mlp_layer_num - 1)))

        # GMF的Embedding层
        self.gmf_user_embedding = torch.nn.Embedding(user_num, hidden_dim)
        self.gmf_item_embedding = torch.nn.Embedding(item_num, hidden_dim)

        mlp_Layers = []
        input_size = int(hidden_dim * (2 ** (self.mlp_layer_num)))
        for i in range(self.mlp_layer_num):
            mlp_Layers.append(torch.nn.Linear(int(input_size), int(input_size / 2)))
            mlp_Layers.append(torch.nn.Dropout(self.dropout))
            mlp_Layers.append(torch.nn.ReLU())
            input_size /= 2
        self.mlp_layers = torch.nn.Sequential(*mlp_Layers)
        """
        Sequential(
          (0): Linear(in_features=256, out_features=128, bias=True)
          (1): Dropout(p=0.5, inplace=False)
          (2): ReLU()
          (3): Linear(in_features=128, out_features=64, bias=True)
          (4): Dropout(p=0.5, inplace=False)
          (5): ReLU()
          (6): Linear(in_features=64, out_features=32, bias=True)
          (7): Dropout(p=0.5, inplace=False)
          (8): ReLU()
          (9): Linear(in_features=32, out_features=16, bias=True)
          (10): Dropout(p=0.5, inplace=False)
          (11): ReLU()
        )
        """

        self.output_layer = torch.nn.Linear(2 * self.hidden_dim, 1)

    def forward(self, user, item):
        user_gmf_embedding = self.gmf_user_embedding(user)
        item_gmf_embedding = self.gmf_item_embedding(item)

        user_mlp_embedding = self.mlp_user_embedding(user)
        item_mlp_embedding = self.mlp_item_embedding(item)

        # GMF执行element-wise product操作
        gmf_output = user_gmf_embedding * item_gmf_embedding

        # MLP块通过堆叠的全连接层+激活函数
        mlp_input = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        # 将GMF和MLP的输出结果concat起来，送入最后的全连接层预测结果，并使用sigmoid函数将输出结果映射到0与1之间
        output = torch.sigmoid(self.output_layer(torch.cat([gmf_output, mlp_output], dim=-1))).squeeze(-1)

        return output

    def predict(self, user, item):
        self.eval()
        # print(user.shape,item.shape)#torch.Size([512]) torch.Size([512, 100])
        with torch.no_grad():
            user_gmf_embedding = self.gmf_user_embedding(user)
            item_gmf_embedding = self.gmf_item_embedding(item)

            user_mlp_embedding = self.mlp_user_embedding(user)
            item_mlp_embedding = self.mlp_item_embedding(item)

            gmf_output = user_gmf_embedding.unsqueeze(1) * item_gmf_embedding

            user_mlp_embedding = user_mlp_embedding.unsqueeze(1).expand(-1, item_mlp_embedding.shape[1],
                                                                        -1)  # [512, 128]->[512,1,128]->[512,100,128]
            mlp_input = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=-1)
            mlp_output = self.mlp_layers(mlp_input)

        output = torch.sigmoid(self.output_layer(torch.cat([gmf_output, mlp_output], dim=-1))).squeeze(-1)
        return output


model = NCFModel(hidden_dim, traindataset.user_nums, traindataset.book_nums).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = torch.nn.BCELoss()

loss_for_plot = []
hits_for_plot = []

for epoch in range(epochs):
    # 训练
    losses = []
    for index, data in enumerate(tqdm.tqdm(trainloader)):
        user, item, label = data
        user, item, label = user.to(device), item.to(device), label.to(device).float()
        y_ = model(user, item).squeeze()

        loss = crit(y_, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())

    # 验证
    hits = []
    for index, data in enumerate(validloader):
        user, pos, neg = data
        # print(pos.shape,neg.shape)#torch.Size([512]) torch.Size([512, 99])
        pos = pos.unsqueeze(1)  # [512->[512,1]
        all_data = torch.cat([pos, neg], dim=-1)
        print(all_data)  # torch.Size([512, 100])
        output = model.predict(user.to(device), all_data.to(device)).detach().cpu()  ##torch.Size([512, 100])

        # 每一个用户预测的结果对应output中的一行(batch),被预测的相应item是all_data中的一行(batch_items)
        for batch, batch_items in zip(output, all_data):

            pos_id = batch_items[0]  # 取出正样本对应的真实的item id

            pred10 = (batch).argsort(descending=True)[:10]  # 预测值从大到小，取前10所在下标
            pred10 = batch_items[pred10]  # 在batch_items中的真实下标，这才是item id
            print(pred10)
            # 索引0是正样本，如果预测的前10中没有0，那么说明预测错了
            if pos_id not in pred10:
                hits.append(0)
            else:
                hits.append(1)

    print('Epoch {} finished, average loss {}, hits@20 {}'.format(epoch, sum(losses) / len(losses),
                                                                  sum(hits) / len(hits)))
    loss_for_plot.append(sum(losses) / len(losses))
    hits_for_plot.append(sum(hits) / len(hits))

# 模型保存
torch.save(model.state_dict(), 'D:/大三上/机器学习/课程设计/Book _Recommendations/model.h5')

import matplotlib.pyplot as plt

x = list(range(1, len(hits_for_plot) + 1))
plt.subplot(1, 2, 1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(x, loss_for_plot, 'r')

plt.subplot(1, 2, 2)
plt.xlabel('epochs')
plt.ylabel('acc')
plt.plot(x, hits_for_plot, 'r')

plt.show()

df = pd.read_csv("D:/大三上/机器学习/课程设计/Book _Recommendations/test_dataset.csv")
user_for_test = df['user_id'].tolist()

predict_item_id = []


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i, i + n]


f = open('D:/大三上/机器学习/课程设计/Book _Recommendations/submission.csv', 'w', encoding='utf-8')
predict_item_id = []
# 预测每一个用户user可能会点击的图书item
for user in tqdm.tqdm(user_for_test):
    # 将用户已经交互过的物品排除
    user_visited_items = traindataset.user_book_map[user]
    items_for_predict = list(set(range(traindataset.book_nums)) - set(user_visited_items))

    results = []
    user = torch.Tensor([user]).to(device).long()

    for item_batch in chunks(items_for_predict, 512):
        item_batch = torch.Tensor(item_batch).unsqueeze(0).to(device).long()

        # print(user.shape)#torch.Size([1])
        # print(item_batch.shape)#torch.Size([1, 512])

        result = model.predict(user, item_batch).view(-1).detach().cpu()
        # print(result.shape)#torch.Size([512])
        results.append(result)
    # print(len(results),len(results[0]))#(20,512),注意results[-1]不一定是512，因为可能不足一个batch (batch size =512)
    results = torch.cat(results, dim=-1)  # 所有items_for_predict关于用户user的预测值

    # 取得分前10的item在results(也在items_for_predict)中的下标
    predict_item_id = results.argsort(descending=True)[:10]  # 从大到小排序，取前10
    print('ind:', predict_item_id)

    # 映射到真实的item id
    res = []
    for i in predict_item_id:
        res.append(items_for_predict[i])
    print('res:', res)
    list(map(lambda x: f.write('{},{}\n'.format(user.cpu().item(), x)), predict_item_id))

f.flush()
f.close()