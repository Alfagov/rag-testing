#%%
from flask import Flask, request, jsonify
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from setup_llm import load_llm, embed_model
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever, RecursiveRetriever
from llama_index.core.node_parser import CodeSplitter, MarkdownNodeParser
from llama_index.core import load_index_from_storage, get_response_synthesizer,StorageContext, ServiceContext, VectorStoreIndex

#%%

app = Flask(__name__)

def setup_rag():
    llm = load_llm()
    print("Successfully loaded LLM")
    
    sctx = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
    )
    
    docstore = SimpleDocumentStore.from_persist_path("./docstore")
    st_ctx = StorageContext.from_defaults(docstore=docstore)
    
    am_indexd = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./index/"),
        service_context=sctx,
    )
    
    
    retriver = am_indexd.as_retriever(
        similarity_top_k=5,
    )
    
    rec_retriver = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": retriver},
        verbose=True,
    )
    
    response_synthesizer = get_response_synthesizer(service_context=sctx, verbose=True)
    rerank = SentenceTransformerRerank(top_n=5, model="BAAI/bge-reranker-base")
    
    engine = RetrieverQueryEngine.from_args(
        rec_retriver, llm=llm, response_synthesizer=response_synthesizer, rerank=rerank
    )
    
    return engine

@app.route('/query', methods=['POST'])
def handle_query():
    data_in = request.get_json()
    response = engine.query(data_in['query'])
    return jsonify(str(response))

if __name__ == '__main__':
    engine = setup_rag()
    app.run(host="0.0.0.0", port=5000, debug=True)
