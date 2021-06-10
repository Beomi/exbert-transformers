<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# exBERT on Transformers 🤗

## Original exBERT

- Repo: https://github.com/cgmhaicenter/exBERT
- Paper: https://www.aclweb.org/anthology/2020.findings-emnlp.129/

## Updated for Transformers 🤗

- PyTorch 1.8.1 ✅
- Huggingface Trainer ✅
- AutoModel, AutoTokenizer ✅
- DeepSpeed Pretrain with `run_mlm.py` ✅
- GPU ✅ (TPU test in progress)
- Fine tune available (https://github.com/Beomi/KcBERT-finetune, In progress)

## How to use

### Pretrain exBERT

- Need to clone this repo

```sh
git clone https://github.com/Beomi/exbert-transformers
cd exbert-transformers
pip install -e ".[dev]" && pip install datasets
cd examples/pytorch/language-modeling/
./exbert_pretrain.sh
```

### Finetune

**Install exbert-transformers**

- No need to git clone repo!

```sh
pip install git+https://github.com/Beomi/exbert-transformers
```

**Load**

```python
from transformers import exBertModel, exBertTokenizer

model = exBertModel.from_pretrained(...)
tokenizer = exBertTokenizer.from_pretrained(...)
```

**Trained on PAWS**

```python
from transformers import exBertModel, exBertTokenizer

model = exBertModel.from_pretrained('beomi/exKcBERT-paws')
tokenizer = exBertTokenizer.from_pretrained('beomi/exKcBERT-paws')
```

> Note) The `base_model` of Finetuned model config should be `""`(blank)


## Vocab update

If you want to change base BERT model or add more vocab on exBERT, add vocab or update vocab on `examples/pytorch/language-modeling/exbert/vocab.txt` 
and update `vocab_size` and `base_model`  on `examples/pytorch/language-modeling/exbert/config.json`.

## Appendix

### Sample Train result example

Terminal results on Github GIST: https://gist.github.com/Beomi/1aa650f75c8e9b3dd467038004244ed2



