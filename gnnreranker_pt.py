import torch
import numpy as np
import pandas as pd
import pyterrier as pt
from torch_geometric.data import Data
from corpus_graph import NpTopKCorpusGraph
from train3 import GNNReRanker  

class GNNScorer(pt.Transformer):
    def __init__(self,
                 model_path: str,
                 graph_path: str,
                 embeddings_path: str,
                 device: str = "cuda"):

        self.device = torch.device(device)
        self.graph = NpTopKCorpusGraph(graph_path)
        
        self.embeddings = torch.load(embeddings_path, map_location=self.device)
        emb_dim = self.embeddings.size(1)
        self.model = GNNReRanker( 
            in_dim=emb_dim,
            hidden_dim=256,
            num_layers=2
        )

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        docnos = df['docno'].tolist()
        inv = self.graph._docnos.inv
        batch_ids = [inv[d] for d in docnos]

        neigh = self.graph.edges_data[batch_ids]   
        all_ids = np.unique(np.concatenate([batch_ids, neigh.flatten()]))
        id2sub = {nid:i for i,nid in enumerate(all_ids)}

        x = self.embeddings[all_ids].to(self.device) 

        src_list, dst_list = [], []
        for src in all_ids:
            for dst in self.graph.edges_data[src]:
                if dst in id2sub:
                    src_list.append(id2sub[src])
                    dst_list.append(id2sub[dst])
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=self.device)

        batch_pos = torch.tensor([id2sub[src] for src in batch_ids], device=self.device)

        with torch.no_grad():
            scores = self.model(x, edge_index, batch_pos) 

        out = df.copy()
        out['score'] = scores.cpu().numpy()
        return out
