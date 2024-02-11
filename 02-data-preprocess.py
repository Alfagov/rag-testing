#!/usr/bin/env python
# coding: utf-8

# In[1]:


from llama_index import SimpleDirectoryReader
from llama_index.ingestion import IngestionPipeline
from llama_index.node_parser import CodeSplitter, TokenTextSplitter, MarkdownNodeParser
from llama_index.extractors import TitleExtractor, QuestionsAnsweredExtractor, EntityExtractor
from setup_llm import load_llm


# In[2]:


# Load raw data from directory
code_documents = SimpleDirectoryReader(
    input_dir="./input",
    required_exts=[".go"],
    recursive=True,
).load_data()

md_documents = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".md"],
    recursive=True,
).load_data()

print(f"Loaded {len(code_documents)} documents")
print(f"Loaded {len(md_documents)} documents")


# In[4]:


# Create parser for code and txt
code_parser = CodeSplitter.from_defaults(
    language="go",
)

md_parser = MarkdownNodeParser.from_defaults()


# In[2]:


llm = load_llm()


# In[5]:


# Define the metadata extractors
title_extractor = TitleExtractor(llm=llm)
qa_extractor = QuestionsAnsweredExtractor(llm=llm, questions=3)
en_extractor = EntityExtractor()


# In[6]:


# Define the ingestion pipeline
pipeline_code = IngestionPipeline(
    transformations=[code_parser, title_extractor,  en_extractor],
)

pipeline_txt = IngestionPipeline(
    transformations=[md_parser, title_extractor, qa_extractor, en_extractor],
)


# In[ ]:


nodes_code = pipeline_code.run(
    documents=code_documents,
    in_place=True,
    show_progress=True,
)
nodes_txt = pipeline_txt.run(
    documents=md_documents,
    in_place=True,
    show_progress=True,
)


# In[ ]:


import pickle

pickle.dump(nodes_code, open("./preproc_nodes/fiber_nodes_code.pkl", "wb"))
pickle.dump(nodes_txt, open("./preproc_nodes/fiber_nodes_txt.pkl", "wb"))


# In[5]:


from llama_index import ServiceContext, VectorStoreIndex, StorageContext

auto_merging_context_c = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    node_parser=code_parser,
)

auto_merging_context_t = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    node_parser=md_parser,
)

storage_context_c = StorageContext.from_defaults()
storage_context_c.docstore.add_documents(nodes_code)

storage_context_t = StorageContext.from_defaults()
storage_context_t.docstore.add_documents(nodes_txt)

automerging_index = VectorStoreIndex(
    nodes=nodes_code,
    storage_context=storage_context_c,
    service_context=auto_merging_context_c,
)

automerging_index_t = VectorStoreIndex(
    nodes=nodes_txt,
    storage_context=storage_context_t,
    service_context=auto_merging_context_t,
)

automerging_index.storage_context.persist(persist_dir="./preproc_data/fiber_merging_index_c")
automerging_index_t.storage_context.persist(persist_dir="./preproc_data/fiber_merging_index_t")


# In[ ]:




