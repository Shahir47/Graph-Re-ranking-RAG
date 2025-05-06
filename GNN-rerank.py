import os
import torch
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from sentence_transformers import SentenceTransformer
import ir_datasets
from corpus_graph import NpTopKCorpusGraph
import pandas as pd
from tqdm import tqdm


EMBED_DIM = 384
HIDDEN_DIM = 256
NUM_LAYERS = 2  
EPOCHS = 3
LEARNING_RATE = 1e-3


dataset_ = ir_datasets.load('msmarco-passage/train')
passages = {}

for doc in tqdm(dataset_.docs_iter(), desc='Loading passages...'):
    passages[doc.doc_id] = doc.text


print("----------------------------- computing passage embeddings -----------------------------")
cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device = cuda_device)

passage_ids = list(passages.keys())
passage_texts = [passages[pid] for pid in passage_ids]

embeddings = []
BATCH_SIZE = 1024

for i in tqdm(range(0, len(passage_texts), BATCH_SIZE), desc='Embedding passages...'):
    batch_embedding = embedder.encode(passage_texts[i : i+BATCH_SIZE], convert_to_tensor=True)
    embeddings.append(batch_embedding.cpu())

embeddings = torch.cat(embeddings, dim=0) 


#passage_id to index mapping
pid2idx = {pid: idx for idx, pid in enumerate(passage_ids)}


print("----------------------------- Loading precomputed corpus graphs -----------------------------")
graph = NpTopKCorpusGraph('./corpusgraph_bm25_k8')
edge_index = torch.tensor(graph.edges_data.T, dtype=torch.long) 

#PyG Data object
x = embeddings

labels = torch.zeros(x.size(0), dtype=torch.float)
for qrel in tqdm(ds.qrels_iter(), desc='Building labels...'):
    pid = qrel.passage_id
    if pid in pid2idx:
        labels[pid2idx[pid]] = 1.0

data = Data(x=x, edge_index=edge_index)

# NeighborLoader for mini-batch training
loader = NeighborLoader(data, num_neighbors= [8, 8], batch_size=BATCH_SIZE, input_nodes=torch.arange(data.num_nodes))

class GNNReRanker(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.lin(x[batch]).squeeze(-1)

model = GNNReRanker(EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(cuda_device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()


print("----------------------------- Training -----------------------------")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0
    loader_iter = tqdm(loader, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch')
    
    for batch in loader_iter:
        batch = batch.to(cuda_device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        batch_labels = labels[batch.batch].to(cuda_device)
        loss = criterion(out, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch.size(0)
        loader_iter.set_postfix(loss=loss.item())

    avg_loss = total_loss / data.num_nodes
    print(f'Epoch {epoch+1}/{EPOCHS} completed. Avg Loss: {avg_loss:.4f}')


print("----------------------------- Saving model -----------------------------")
torch.save(model.state_dict(), 'gnn_reranker.pth')
print('Training complete. Model saved as gnn_reranker.pth')