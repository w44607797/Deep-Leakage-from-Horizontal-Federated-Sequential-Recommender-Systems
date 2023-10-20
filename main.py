import argparse
from tqdm import tqdm
import torch.nn as nn
import torch

from SimpleRNNWithEmbedding import SimpleRNNWithEmbedding

parser = argparse.ArgumentParser()
parser.add_argument("--sequence_length", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--input_size", type=int, default=2000, help="number of the item")
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--output_size", type=int, default=16)
parser.add_argument("--hidden_size", type=int, default=32)
parser.add_argument("--noise_stddev", type=float, default=0, help="noise standard deviation")
parser.add_argument("--num_initializations", type=int, default=20)
parser.add_argument("--train_epoch", type=int, default=300)
parser.add_argument("--device", type=str, default="cuda")


args = parser.parse_args()

random_data = torch.randint(1, args.input_size, (args.batch_size, args.sequence_length)).to(args.device)

criterion = nn.MSELoss()
embedding = nn.Embedding(args.input_size, args.embedding_dim).to(args.device)
model = SimpleRNNWithEmbedding(hidden_size = args.hidden_size ,output_size = args.output_size ,embedding_dim = args.embedding_dim ).to(args.device)


input_data = random_data[:,:args.sequence_length-1]
label_data = random_data[:,-(args.sequence_length-1):]


input = embedding(input_data.to(args.device))
label_embed = embedding(label_data.to(args.device))
out = model(input)
loss = criterion(out, label_embed).mean()

dy_dx = torch.autograd.grad(loss, model.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))


noisy_gradients = []
for grad in original_dy_dx:
    noise = torch.normal(mean=0.0, std=args.noise_stddev, size=grad.size(), device=grad.device)
    noisy_grad = grad + noise
    noisy_gradients.append(noisy_grad)

if args.noise_stddev != 0:
    original_dy_dx = noisy_gradients




best_loss = float('inf')
best_data = None
best_label = None
dummy_data_set = []
loss_sets = []

train_epoch = args.train_epoch
for init_num in range(args.num_initializations):
    # Randomly initialize your dummy_data and dummy_label
    dummy_data = torch.randint(1, args.input_size, (args.batch_size, args.sequence_length)).to(args.device)
    input_dummy_data = dummy_data[:,:args.sequence_length-1]
    label_dummy_data = dummy_data[:,-(args.sequence_length-1):]
    dummy_data = embedding(input_dummy_data).detach().requires_grad_(True)
    dummy_label = embedding(label_dummy_data.to(args.device)).detach().requires_grad_(True)
    # print(f"initializations No:{init_num} Processing")
    loss_set = []
    for iters in tqdm(range(1,train_epoch+1), desc=f"initializations No:{init_num} Processing"):
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
        with torch.no_grad():
            dummy_label[:,:-1,:] = dummy_data[:,1:,:]
        dummy_label.requires_grad_(True)
        dummy_data.requires_grad_(True)

        def closure():
            optimizer.zero_grad()
            with torch.backends.cudnn.flags(enabled=False):
                dummy_pred = model(dummy_data)
            dummy_loss = criterion(dummy_pred, dummy_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx-gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff
        loss_set.append(closure().item())
        optimizer.step(closure)
        if iters % (train_epoch) == 0 and iters!=0:
            current_loss = closure()
            dummy_data_set.append(dummy_data)
            print(f"Initialization {init_num + 1}/{args.num_initializations}, Iteration {iters}, Loss: %.16f" % current_loss.item())

    loss_sets.append(loss_set)
    final_loss = closure()
    # Check if the current initialization yields a better loss
    if final_loss < best_loss:
        best_loss = final_loss
        best_data = dummy_data.clone().detach()
        best_label = dummy_label.clone().detach()

print("Best Loss after all initializations: %.16f" % best_loss.item())

ew = embedding.weight
# 计算欧氏距离
# print("predict")
nearest_indices_sets = []
nearest_indices2_sets = []
def get_indices_sets(predict,origin):
    nearest_indices_set=[]
    for i in range(args.batch_size):
        distance_matrix = torch.cdist(predict[i], ew)
        nearest_indices = torch.argmin(distance_matrix, dim=1)
        nearest_indices_set.append(nearest_indices)
        # print(nearest_indices)
    # print('-------------------------')
    # print("label")
    nearest_indices_set2=[]
    for i in range(args.batch_size):
        distance_matrix2 = torch.cdist(origin[i], ew)
        nearest_indices2 = torch.argmin(distance_matrix2, dim=1)
        nearest_indices_set2.append(nearest_indices2)
        # print(nearest_indices2)
    return nearest_indices_set,nearest_indices_set2

for i in tqdm(range(0,args.num_initializations)):
    nearest_indices_set,nearest_indices_set2 = get_indices_sets(dummy_data_set[i],input)
    nearest_indices_sets.append(nearest_indices_set)
    nearest_indices2_sets.append(nearest_indices_set2)


import torch
def valid(nearest_indices_set,nearest_indices_set2,num_initializations):
    def sequence_similarity(seq1, seq2):
        return (seq1 == seq2).float().mean().item()

    def find_best_match(seq, seq_set):
        return max([sequence_similarity(seq, s) for s in seq_set])


    # 对于seq_set_1中的每个序列找到最佳的匹配
    similarities = [find_best_match(seq, nearest_indices_set2) for seq in nearest_indices_set]
    # 计算平均相似度
    average_similarity = sum(similarities) / len(similarities)

    print(f"第{num_initializations}次预测泄露程度: {average_similarity * 100:.2f}%")
    return f"{average_similarity * 100:.2f}%"
#%%
score_sets = []
for i in range(args.num_initializations):
    score_sets.append(valid(nearest_indices_sets[i],nearest_indices2_sets[i],i))


import matplotlib.pyplot as plt

x = list(range(1, len(loss_sets[0])+1))


for i, column in enumerate(loss_sets):
    plt.plot(x, column, label=f'LR {score_sets[i]}')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
