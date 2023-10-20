import torch
from torch import nn


class SimpleRNNWithEmbedding(nn.Module):
    def __init__(self,  hidden_size, output_size, embedding_dim):
        super(SimpleRNNWithEmbedding, self).__init__()
        self.hidden_size = hidden_size

        # RNN层
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, embedded):
        # 将输入序列通过嵌入层映射为嵌入向量
        # embedded = self.embedding(x)

        # 初始化隐状态
        h0 = torch.zeros(1, embedded.size(0), self.hidden_size).to(embedded.device)

        # 前向传播
        out, _ = self.rnn(embedded, h0)

        # 提取RNN最后一层的输出并送入全连接层
        out = self.fc(out)

        return out
