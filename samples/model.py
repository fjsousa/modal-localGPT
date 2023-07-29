from modal import Image, Stub, gpu, method, web_endpoint, NetworkFileSystem
import torch

model_id = "TheBloke/vicuna-7B-1.1-HF"
cache_path = "/vol/cache"

def image_setup_with_model_local_save():
    from transformers import AutoModelForCausalLM
    import time

    start = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,  
        torch_dtype=torch.float16,
        #low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=cache_path,
        #subfolder="automodel",
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
    
    model.save_pretrained(cache_path, safe_serialization=True)
    end = time.time()
    print("saved model into image", end - start)

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
    .run_function(image_setup_with_model_local_save, gpu="any"))


stub = Stub(name="just-model-local-save-2", image=image)

@stub.cls(gpu="any", timeout=5000)
class Model:


    def  __enter__(self):
        import time
        from transformers import AutoModelForCausalLM, AutoConfig
        start = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            #subfolder="automodel",
            low_cpu_mem_usage=True,   
            torch_dtype=torch.float16,
            #low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        end = time.time()
        print("AutoModelForCausalLM at __enter__...", end - start)

    #loads model and replies "ok"
    @method()
    def echo(self):
        bla = self.model
        return "ok"

@stub.local_entrypoint()
def cli():
    import time
    start = time.time()
    Model().echo.call()
    end = time.time()
    print("cli load time...", end - start)

@stub.function()
@web_endpoint()
def get():
    import time
    start = time.time()
    Model().echo.call()
    end = time.time()
    print("get load time", end - start)