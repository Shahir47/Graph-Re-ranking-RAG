import torch
import ir_datasets
from tqdm import tqdm
from corpus_graph import NpTopKCorpusGraph
from sentence_transformers import SentenceTransformer

graph = NpTopKCorpusGraph('./corpusgraph_bm25_k8')

dataset_ = ir_datasets.load('msmarco-passage/train')
passages = {}

for doc in tqdm(dataset_.docs_iter(), desc='Loading passages...'):
    passages[doc.doc_id] = doc.text

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

embedddings_dt = {docno: doc_vec for docno, doc_vec in zip(passage_ids, embeddings)}

rearranged_embeddings = torch.stack([embedddings_dt[docno] for docno in graph._docnos.fwd])

torch.save(rearranged_embeddings, "embeddings_aligned.pt")

print("Done!!!")