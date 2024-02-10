# %%
from llama_index import SimpleDirectoryReader
from llama_index.ingestion import IngestionPipeline
from llama_index.node_parser import CodeSplitter, TokenTextSplitter
from llama_index.extractors import TitleExtractor, QuestionsAnsweredExtractor, EntityExtractor
from setup_llm import load_llm

# %%
# Load raw data from directory
code_documents = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".go"],
    recursive=True,
).load_data()

txt_documents = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".md"],
    recursive=True,
).load_data()

print(f"Loaded {len(code_documents)} documents")
print(f"Loaded {len(txt_documents)} documents")

# %%
# Create parser for code and txt
code_parser = CodeSplitter.from_defaults(
    language="go",
)

txt_parser = TokenTextSplitter.from_defaults()

# %%
llm = load_llm()

# %%
# Define the metadata extractors
title_extractor = TitleExtractor(llm=llm)
qa_extractor = QuestionsAnsweredExtractor(llm=llm, questions=3)
en_extractor = EntityExtractor()

# %%
# Define the ingestion pipeline
pipeline_code = IngestionPipeline(
    transformations=[code_parser, title_extractor, qa_extractor, en_extractor],
)

# %%
import pickle
nodes_code = pipeline_code.run(
    documents=code_documents,
    show_progress=True,
)

pickle.dump(nodes_code, open("nodes_code.pkl", "wb"))

# %%
import pickle
from llama_index import ServiceContext, VectorStoreIndex, StorageContext

nodes_code = pickle.load(open("nodes_code.pkl", "rb"))

auto_merging_context_c = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    node_parser=code_parser,
)

storage_context_c = StorageContext.from_defaults()
storage_context_c.docstore.add_documents(nodes_code)

automerging_index = VectorStoreIndex(
    nodes=nodes_code,
    storage_context=storage_context_c,
    service_context=auto_merging_context_c,
)

automerging_index.storage_context.persist(persist_dir="./merging_index_c_lg")

# %%



