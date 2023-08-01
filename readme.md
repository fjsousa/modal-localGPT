Based on [localGPT](https://github.com/PromtEngineer/localGPT) but only for Llama-7B atm.

Runs on [modal.com](https://modal.com).
```
model_id = "TheBloke/Llama-2-7B-Chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
```

## Run/deploy

```
modal deploy/run localgpt.py
```

## Try it

```
$ time curl -sG https://fjsousa--localgpt-superconductor-get.modal.run --data-urlencode 'question=based on the paper I provided whats lk-99' | jq ".answer"

# Llama-2-7B
" Based on the paper you provided, LK-99 is a new room-temperature superconductor material that has been discovered by the authors of the paper. It is a complex compound made up of several elements, including lead, copper, and phosphorus, and it exhibits superconductivity at room temperature and ambient pressure. The authors of the paper propose that the superconductivity in LK-99 is due to the presence of correlated SQWs (structural quantum wedges) in the material, which are generated by an appropriate amount of distortion at room temperature and ambient pressure without any relaxation process."
real	1m53.787s

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
