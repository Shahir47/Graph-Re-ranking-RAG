import pyterrier as pt
pt.init()
from pyterrier_pisa import PisaIndex
from corpus_graph import CorpusGraph


def main():
    GRAPH_DIR = './corpusgraph_bm25_k8'  
    K = 8                                 
    BATCH_SIZE = 1024                     


    dataset = pt.get_dataset('irds:msmarco-passage')
    index = PisaIndex.from_dataset('msmarco_passage')
    retriever = index.bm25()

    print(f"Building BM25 graph (k={K}) in '{GRAPH_DIR}'...")

    docs_iter = dataset.get_corpus_iter()
    CorpusGraph.from_retriever(retriever, docs_iter, GRAPH_DIR, k=K, batch_size=BATCH_SIZE)
    
    print("Graph build complete. Check for:")
    print(f"  {GRAPH_DIR}/pt_meta.json")
    print(f"  {GRAPH_DIR}/edges.u32.np")
    print(f"  {GRAPH_DIR}/weights.f16.np")

if __name__ == '__main__':
    main()