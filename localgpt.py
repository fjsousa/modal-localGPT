import os
from modal import Image, Stub, gpu, method, web_endpoint, NetworkFileSystem, Mount

from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from langchain.vectorstores import Chroma

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

IMAGE_MODEL_DIR = "/model"

volume = NetworkFileSystem.persisted("model-cache-vol")

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

# def download_model():
#     from huggingface_hub import snapshot_download

#     model_name = "TheBloke/falcon-40b-instruct-GPTQ"
#     snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR)

def presist_db_run_model():
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

    print(">>>>about to presist")
    db.persist()
    db = None


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
    .run_function(presist_db_run_model, gpu="any", mounts=[Mount.from_local_dir("SOURCE_DOCUMENTS", remote_path="/root/SOURCE_DOCUMENTS"), Mount.from_local_dir("DB", remote_path="/root/DB")]))

stub = Stub(name="localgpt-db", image=image)

CACHE_DIR = "/DB"

# @stub.cls(gpu=gpu.A100(), timeout=60 * 10, container_idle_timeout=60 * 5)
# class Falcon40BGPTQ:
#     def __enter__(self):
#         from transformers import AutoTokenizer
#         from auto_gptq import AutoGPTQForCausalLM

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             IMAGE_MODEL_DIR, use_fast=True
#         )
#         print("Loaded tokenizer.")

#         self.model = AutoGPTQForCausalLM.from_quantized(
#             IMAGE_MODEL_DIR,
#             trust_remote_code=True,
#             use_safetensors=True,
#             device_map="auto",
#             use_triton=False,
#             strict=False,
#         )
#         print("Loaded model.")

#     @method()
#     def generate(self, prompt: str):
#         from threading import Thread
#         from transformers import TextIteratorStreamer

#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         streamer = TextIteratorStreamer(
#             self.tokenizer, skip_special_tokens=True
#         )
#         generation_kwargs = dict(
#             inputs=inputs.input_ids.cuda(),
#             attention_mask=inputs.attention_mask,
#             temperature=0.1,
#             max_new_tokens=512,
#             streamer=streamer,
#         )

#         # Run generation on separate thread to enable response streaming.
#         thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
#         thread.start()
#         for new_text in streamer:
#             yield new_text

#         thread.join()

# prompt_template = (
#     "A chat between a curious human user and an artificial intelligence assistant. The assistant give a helpful, detailed, and accurate answer to the user's question."
#     "\n\nUser:\n{}\n\nAssistant:\n"
# )


@stub.local_entrypoint()
def cli():
    question = "who painted the gioconda"
    #model = Falcon40BGPTQ()
    print("runnin model")
    #for text in model.generate.call(prompt_template.format(question)):
    #    print(text, end="", flush=True)

#@stub.function(timeout=60 * 10)
#@web_endpoint()
# def get(question: str):
#     from fastapi.responses import StreamingResponse
#     from itertools import chain

#     model = Falcon40BGPTQ()
#     return StreamingResponse(
#         chain(
#             ("Loading model. This usually takes around 20s ...\n\n"),
#             model.generate.call(prompt_template.format(question)),
#         ),
#         media_type="text/event-stream",
    
