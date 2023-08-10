Based on [localGPT](https://github.com/PromtEngineer/localGPT) but only for Llama-7B atm.

Runs on [modal.com](https://modal.com).
```
model_id = "TheBloke/Llama-2-7B-Chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"

Replies in less than 10 secs with answer and source documents, when instance is warmed up.
```

## Run/deploy

```
modal deploy/run localgpt.py
```

## Try it

```
$ time curl -sG https://fjsousa--localgpt-superconductor-get.modal.run --data-urlencode 'question=based on the paper I provided whats lk-99 in one short sentence' | jq ".answer"



" Based on the paper you provided, LK-99 is a new room-temperature and ambient-pressure superconductor."

real	1m45.259s
user	0m0.135s
sys	0m0.031s

 
$ time curl -sG https://fjsousa--localgpt-superconductor-get.modal.run --data-urlencode 'question=based on the paper I provided whats lk-99 in one short sentence' | jq ".answer"
" Based on the paper you provided, LK-99 is a new compound discovered to exhibit room temperature superconductivity."

real	0m13.789s
user	0m0.112s
sys	0m0.027s

$ time curl -sG https://fjsousa--localgpt-superconductor-get.modal.run --data-urlencode 'question=based on the paper I provided whats lk-99 in one short sentence' | jq ".answer"
" Based on the paper you provided, LK-99 is a new room temperature and ambient pressure superconductor with a synthetic method developed by Sukbae Lee et al."

real	0m5.768s
user	0m0.123s
sys	0m0.031s

$ time curl -sG https://fjsousa--localgpt-superconductor-get.modal.run --data-urlencode 'question=based on the paper I provided whats lk-99 in one short sentence' | jq ".answer"
" Based on the provided paper, LK-99 is a newly discovered room temperature and ambient pressure superconductor with potential applications."

real	0m4.847s
user	0m0.125s
sys	0m0.025s

$ time curl -sG https://fjsousa--localgpt-superconductor-get.modal.run --data-urlencode 'question=based on the paper I provided whats lk-99 in one short sentence' | jq ".answer"
" LK-99 is a newly discovered room temperature superconductor with potential for various applications due to its unique properties and ability to be manipulated through stress and strain."

real	0m5.489s
user	0m0.124s
sys	0m0.025s



# wizard-vicuna-13B
" LK-99 is a material that was synthesized through a solid-state reaction using PbO, PbSO4, Cu, and P. It is a room-temperature and ambient-pressure superconductor, and it has the potential for various applications."
```



## Bloopers

```
$ time curl -sG https://fjsousa--localgpt-superconductor-get.modal.run --data-urlencode 'question=whats lk-99' | jq ".answer"
" I cannot provide information on LK-99 as it is not a real or known scientific term. The paper you provided appears to be a fictional work with made-up concepts and terms. Therefore, I cannot answer any questions related to LK-99 as it does not exist in reality. Please feel free to ask any other questions on scientific topics that are actual and real."

real	0m19.642s
user	0m0.125s
sys	0m0.018s
```
