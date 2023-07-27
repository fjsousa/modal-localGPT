import os
import logging
import time
from modal import Image, Stub, gpu, method, web_endpoint, NetworkFileSystem, Mount

import torch

from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

IMAGE_MODEL_DIR = "/model"

model_id = "TheBloke/vicuna-7B-1.1-HF"
model_basename = None

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]

def load_document_batch(filepaths):
    #logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)

def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory
    all_files = os.listdir(source_dir)
    paths = []
    for file_path in all_files:
        file_extension = os.path.splitext(file_path)[1]
        source_file_path = os.path.join(source_dir, file_path)
        if file_extension in DOCUMENT_MAP.keys():
            paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs #, python_docs

VOLUME_DIR = "/root/cache-volume"
TOKENIZER_CACHE_PATH = "/root/cache-volume/tokenizer"
MODEL_CACHE_PATH = "/root/cache-volume/automodel"
GENERATION_CACHE_PATH = "/root/cache-volume/generation"

volume = NetworkFileSystem.persisted("cache_volume")

def presist_db_run_model():
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    from langchain.vectorstores import Chroma
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    import torch
    
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(text_documents)   

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
    )

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )

    db.persist()
    db = None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(TOKENIZER_CACHE_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,  
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
    model.save_pretrained(MODEL_CACHE_PATH)

    generation_config = GenerationConfig.from_pretrained(model_id, use_cache=True)
    generation_config.save_pretrained(GENERATION_CACHE_PATH)

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "langchain==0.0.191",
        "chromadb==0.3.22",
        "pdfminer.six==20221105",
        "InstructorEmbedding",
        "sentence-transformers",
        "faiss-cpu",    
        "huggingface_hub",
        "transformers",
        "protobuf==3.20.0; sys_platform != 'darwin'",
        "protobuf==3.20.0; sys_platform == 'darwin' and platform_machine != 'arm64'",
        "protobuf==3.20.3; sys_platform == 'darwin' and platform_machine == 'arm64'",
        "auto-gptq",
        "docx2txt",
        "urllib3==1.26.6",
        "accelerate",
        "bitsandbytes",
        "click",
        "flask",
        "requests",
        "openpyxl"
    )
    #.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # #Mount.from_local_dir("DB", remote_path="/root/DB")
    .run_function(presist_db_run_model, gpu="any", network_file_systems={VOLUME_DIR: volume}, mounts=[Mount.from_local_dir("SOURCE_DOCUMENTS", remote_path="/root/SOURCE_DOCUMENTS")]))





stub = Stub(name="localgpt-db-with-model", image=image)

@stub.function(gpu="any", network_file_systems={VOLUME_DIR: volume}, timeout=500)
def modal_function():

    from langchain.embeddings import HuggingFaceInstructEmbeddings
    from langchain.vectorstores import Chroma
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig
    import torch
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda"})

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )

    retriever = db.as_retriever()

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CACHE_PATH)
    end = time.time()
    print("setting up Autotokenizer...", end - start)

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_CACHE_PATH, 
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True,
                                                 trust_remote_code=True, config=AutoConfig.from_pretrained(MODEL_CACHE_PATH))
    end = time.time()
    print("AutoModelForCausalLM...", end - start)
    #AutoModelForCausalLM... 67.84899544715881

    model.tie_weights()

    generation_config = GenerationConfig.from_pretrained(GENERATION_CACHE_PATH)  

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, device_map="auto",)
    
    question = "All legislative Powers herein granted shall be vested in a..."


    res = qa(question)
    answer, docs = res["result"], res["source_documents"]

    #print(docs)
    return answer  

@stub.local_entrypoint()
def cli():
    modal_function.call()


#     return StreamingResponse(
#         chain(
#             ("Loading model. This usually takes around 20s ...\n\n"),
#             model.generate.call(prompt_template.format(question)),
#         ),
#         media_type="text/event-stream",

@stub.function()#timeout??
@web_endpoint()
def get():
    res = modal_function.call()
    print(">>>>", res)
    return "Hello world"
