# imports
import pyterrier as pt
if not pt.java.started():
    pt.init()
from pyterrier.measures import *
# from pyterrier_adaptive import GAR, NpTopKCorpusGraph
from gar import GAR
from corpus_graph import NpTopKCorpusGraph
from pyterrier_pisa import PisaIndex
from gnnreranker_pt import GNNScorer

print("Prereq...")

# Create required components
dataset = pt.get_dataset('irds:msmarco-passage')
retriever = PisaIndex.from_dataset('msmarco_passage').bm25()
scorer = pt.text.get_text(dataset, 'text') >> GNNScorer(model_path="gnn_reranker.pth", graph_path="corpusgraph_bm25_k8", embeddings_path="embeddings_aligned.pt", device="cuda")
graph = NpTopKCorpusGraph.from_dataset('msmarco_passage', 'corpusgraph_bm25_k16').to_limit_k(8)
print("prereq done...")

# A simple example
pipeline = retriever >> GAR(scorer, graph) >> pt.text.get_text(dataset, 'text')

print("\nPipeline.Search running...\n")
print(pipeline.search('clustering hypothesis information retrieval'))


print("\nEvaluation running...\n")
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
print(pt.Experiment(
    [retriever, retriever >> scorer, retriever >> GAR(scorer, graph)],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG, MAP(rel=2), R(rel=2)@1000],
    names=['bm25', 'bm25 >> duot5', 'bm25 >> GAR(duot5)']
))
print("\nEvaluation done\n")