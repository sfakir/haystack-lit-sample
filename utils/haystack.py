import streamlit as st
import requests
from utils.config import TWITTER_BEARER

# from haystack.nodes import PromptNode, PromptTemplate
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from pprint import pprint
from haystack.utils import print_answers
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.document_stores import InMemoryDocumentStore

import os
retriever, reader, pipe = None, None, None
doc_dir = "data/gameofthrones"

# cached to make index and models load only at start
@st.cache(hash_funcs={"builtins.CoreBPE": lambda _: None}, show_spinner=False, allow_output_mutation=True)
def start_haystack():    
    # if (pipe is not None):
    #     return pipe
    
    
    document_store = InMemoryDocumentStore(use_bm25=True)
    
    retriever = BM25Retriever(document_store=document_store)
    
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    
    pipe = ExtractiveQAPipeline(reader, retriever)
    
    
    files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
    indexing_pipeline = TextIndexingPipeline(document_store)
    indexing_pipeline.run_batch(file_paths=files_to_index)


    st.session_state["haystack_started"] = True                                                     
    
    return pipe
