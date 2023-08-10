import os
import logging
import time
from modal import Image, Stub, gpu, method, web_endpoint, NetworkFileSystem, Mount

import torch

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from langchain.llms import HuggingFacePipeline, LlamaCpp
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


#model_id = "TheBloke/wizard-vicuna-13B-GGML"
#model_basename = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
model_id = "TheBloke/Llama-2-7B-Chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
#model_id = "TheBloke/orca_mini_3B-GGML"
#model_basename = "orca-mini-3b.ggmlv3.q4_0.bin"
device_type = "cuda"


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

cache_path = "/vol/cache-volume"

def presist_db_run_model():
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    from langchain.vectorstores import Chroma
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    import torch
    from huggingface_hub import hf_hub_download
    
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

    start = time.time()

    hf_hub_download(repo_id=model_id, filename=model_basename, cache_dir=cache_path)

    end = time.time()
    print("saved model onto image", end - start)

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "accelerate",
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
        "llama-cpp-python==0.1.66",
        "auto-gptq==0.2.2",
        "docx2txt",
        "urllib3==1.26.6",
        "bitsandbytes",
        "click",
        "flask",
        "requests",
        "openpyxl"
    ).pip_install("xformers", pre=True)
    #.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # #Mount.from_local_dir("DB", remote_path="/root/DB")
    .run_function(presist_db_run_model, gpu="A10G", mounts=[Mount.from_local_dir("SOURCE_DOCUMENTS", remote_path="/root/SOURCE_DOCUMENTS")]))

stub = Stub(name="localgpt-superconductor", image=image)

def load_model(device_type, model_id, model_basename=None):
    from langchain.llms import LlamaCpp
    from huggingface_hub import hf_hub_download
    """
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    # Only supporting - if model_basename is not None:
        #if ".ggml" in model_basename:
    logging.info("Using Llamacpp for GGML quantized models")

    start = time.time()
    model_path = hf_hub_download(repo_id=model_id, filename=model_basename, cache_dir=cache_path)
    end = time.time()
    print("Loading LLM at __enter__...", end - start)

    max_ctx_size = 2048
    kwargs = {
        "model_path": model_path,
        "n_ctx": max_ctx_size,
        "max_tokens": max_ctx_size,
        "n_gpu_layers": 1000,
        "n_batch": max_ctx_size
    }

    return LlamaCpp(**kwargs)

@stub.cls(gpu="T4", timeout=1500, allow_concurrent_inputs=10, concurrency_limit=5)
class Model:

    def  __enter__(self):
        from langchain.embeddings import HuggingFaceInstructEmbeddings
        from langchain.vectorstores import Chroma
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig
        import torch
        embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda"})

        # 2. Loads vector store
        self.db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )

        self.retriever = self.db.as_retriever()

        # 3. Loads LLM
        start = time.time()
        self.llm = load_model(device_type, model_id=model_id, model_basename=model_basename)
        end = time.time()
        print("Loading LLM...", end - start)

    @method()
    def run_inference(self, question):
        from langchain.llms import HuggingFacePipeline
        from transformers import pipeline
        from langchain.chains import RetrievalQA

        llm = self.llm

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            )

        start = time.time()
        res = qa(question)
        end = time.time()
        print("qa(question) took....", end - start)
        answer, docs = res["result"], res["source_documents"]

        #print(docs)
        return {"answer": answer, "docs": docs}# + "\n" + docs[0].metadata["source"] + "\n" + docs[0].page_content



@stub.local_entrypoint()
def cli():
    import time
    start = time.time()
    question = "All legislative Powers herein granted shall be vested in a..."
    res = Model.run_inference.call(question)
    end = time.time()
    print("cli run time", end - start)
    return res

#     return StreamingResponse(
#         chain(
#             ("Loading model. This usually takes around 20s ...\n\n"),
#             model.generate.call(prompt_template.format(question)),
#         ),
#         media_type="text/event-stream",

@stub.function()#timeout??
@web_endpoint()
def get(question: str):
    import time
    start = time.time()
    #question = "All legislative Powers herein granted shall be vested in a..."

    print("question: ", question)
    res = Model.run_inference.call(question)
    end = time.time()
    print("endpoint run time", end - start)
    return res
