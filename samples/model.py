from modal import Image, Stub, gpu, method, web_endpoint, NetworkFileSystem
import torch

model_id = "TheBloke/vicuna-7B-1.1-HF"

VOLUME_DIR = "/root/cache-volume"
MODEL_CACHE_PATH = "/root/cache-volume/automodel"

volume = NetworkFileSystem.persisted("cache_volume")

def image_setup_with_model_local_save():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,  
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
    
    model.save_pretrained(MODEL_CACHE_PATH)

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
    .run_function(image_setup_with_model_local_save, gpu="any", network_file_systems={VOLUME_DIR: volume}))


stub = Stub(name="just-model-local-save-2", image=image)

@stub.function(gpu="any", network_file_systems={VOLUME_DIR : volume}, timeout=500)
def modal_function():
    import time
    from transformers import AutoModelForCausalLM, AutoConfig

    print(">> model load...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_CACHE_PATH, 
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True,
                                                 trust_remote_code=True, config=AutoConfig.from_pretrained(MODEL_CACHE_PATH)
                                                 )
    end = time.time()
    print("AutoModelForCausalLM...", end - start)
    return "ok"

@stub.local_entrypoint()
def cli():
    print(modal_function.call())

@stub.function()
@web_endpoint()
def get():
    res = modal_function.call()
    return res