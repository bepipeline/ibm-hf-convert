---
base_model:
- ibm-granite/granite-vision-3.3-2b-embedding
language:
- en
license: apache-2.0
tags:
- gguf
- mainframe
- s390x
- z15
- z16
- z17
- big-endian
base_model_relation: quantized
quantized_by: taronaeo
---

# Read our How to [Run on IBM Z & LinuxONE Guide!](https://github.com/ggml-org/llama.cpp/blob/master/docs/build-s390x.md)

<div>
<p style="margin: 0 0 0px 0; margin-top: 0px;">
<em>These models are byte-swapped to run on IBM Z & LinuxONE systems.</em>
</p>

<ul style="margin: 0;">
<li>Model Creator: <a href="https://huggingface.co/ibm-granite">IBM Granite</a></li>
<li>Original Model: <a href="https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding">ibm-granite/granite-vision-3.3-2b-embedding</a></li>
</ul>
</div>

<br>

# Provided Files

Get started on your IBM Z & LinuxONE system

```sh
build/bin/llama-server -hf taronaeo/Granite-Vision-3.3-2B-Embedding-BE-GGUF
```

| Name | Quant Method | Bits | Size | Use Case |
|---|---|---|---|---|

---

# granite-vision-3.3-2b-embedding
**Model Summary:**
Granite-vision-3.3-2b-embedding is an efficient embedding model based on [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b). This model is specifically designed for multimodal document retrieval, enabling queries on documents with tables, charts, infographics, and complex layouts. The model generates ColBERT-style multi-vector representations of pages.
By removing the need for OCR-based text extractions, granite-vision-3.3-2b-embedding can help simplify and accelerate RAG pipelines.

**Evaluations:**
We evaluated granite-vision-3.3-2b-embedding alongside other top colBERT style multi-modal embedding models in the 1B-4B parameter range using two benchmark: [Vidore2](https://github.com/illuin-tech/vidore-benchmark/) and [Real-MM-RAG-Bench](https://huggingface.co/collections/ibm-research/real-mm-rag-bench-67d2dc0ddf2dfafe66f09d34) which aim to specifically address complex multimodal document retrieval tasks.

## **NDCG@5 - ViDoRe V2**
| Collection \ Model                     | ColPali-v1.3 | ColQwen2.5-v0.2 | ColNomic-3b |  ColSmolvlm-v0.1     |  granite-vision-3.3-2b-embedding |
|----------------------------------------|--------------|------------------|-------------|-------------------|-----------
| ESG Restaurant Human                   | 51.1        | 68.4           | 65.8       |    62.4               | 65.3                    |
| Economics Macro Multilingual           | 49.9        | 56.5            | 55.4       |     47.4              | 51.2                    |
| MIT Biomedical                         | 59.7        | 63.6            | 63.5       |    58.1               |61.5                   |
| ESG Restaurant Synthetic               | 57.0        | 57.4            | 56.6       |     51.1              |56.6                    |
| ESG Restaurant Synthetic Multilingual  | 55.7        | 57.4            | 57.2       |     47.6             |55.7                    |
| MIT Biomedical Multilingual            | 56.5        | 61.1            | 62.5       |      50.5             | 55.5                    |
| Economics Macro                        | 51.6        | 59.8            | 60.2       |      60.9            |58.3                    |
| **Avg (ViDoRe2)**                      | **54.5**    | **60.6**        | **60.2**   | **54.0**              |**57.7**                    |

## **NDCG@5 - REAL-MM-RAG**
| Collection \ Model                     | ColPali-v1.3 | ColQwen2.5-v0.2 | ColNomic-3b |   ColSmolvlm-v0.1            |  granite-vision-3.3-2b-embedding |
|----------------------------------------|--------------|------------------|-------------|--------------------------| ------------------
| FinReport                              | 55         | 66             | 78        |   65                  |73 
| FinSlides                              | 68        | 79             | 81        |   55                 |79  
| TechReport                             | 78         | 86             | 88        |   83                 |87  
| TechSlides                             | 90         | 93             | 92        |   91            |93   
| **Avg (REAL-MM-RAG)**                  | **73**     | **81**         | **85**    |   **74**           |**83**    

- **Release Date**: June 11th 2025
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Supported Input Format:** Currently the model supports English instructions and images (png, jpeg) as input format.

**Intended Use:**
The model is intended to be used in enterprise applications that involve retrieval of visual and text data. In particular, the model is well-suited for multi-modal RAG systems where the knowledge base is composed of complex enterprise documents, such as reports, slides, images, canned doscuments, manuals and more. The model can be used as a standalone retriever, or alongside a text-based retriever.

### Usage
```shell
pip install -q torch torchvision torchaudio
pip install transformers==4.50
```
Then run the code:
```python
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from transformers.utils.import_utils import is_flash_attn_2_available

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ibm-granite/granite-vision-3.3-2b-embedding"
model = AutoModel.from_pretrained(
                      model_name,
                      trust_remote_code=True,
                      torch_dtype=torch.float16,
                      device_map=device,
                      attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None
                      ).eval()
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# ─────────────────────────────────────────────
# Inputs: Image + Text
# ─────────────────────────────────────────────
image_url = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg"
print("\nFetching image...")
image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")

text = "A photo of a tiger"
print(f"Image and text inputs ready.")

# Process both inputs
print("Processing inputs...")
image_inputs = processor.process_images([image])
text_inputs = processor.process_queries([text])

# Move to correct device
image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

# ─────────────────────────────────────────────
# Run Inference
# ─────────────────────────────────────────────
with torch.no_grad():
    print("🔍 Getting image embedding...")
    img_emb = model(**image_inputs)

    print("✍️ Getting text embedding...")
    txt_emb = model(**text_inputs)

# ─────────────────────────────────────────────
# Score the similarity
# ─────────────────────────────────────────────
print("Scoring similarity...")
similarity = processor.score(txt_emb, img_emb, batch_size=1, device=device)

print("\n" + "=" * 50)
print(f"📊 Similarity between image and text: {similarity.item():.4f}")
print("=" * 50)
```
### Use granite-vision-embedding-3.3-2b for MM RAG
For an example of MM-RAG using granite-vision-3.3-2b-embedding refer to [this notebook](https://github.com/ibm-granite/granite-vision-models/blob/main/cookbooks/GraniteVisionEmbedding_MM-RAG_Notebook.ipynb).

**Model Architecture:**
The architecture of granite-vision-3.3-2b-embedding follows ColPali(https://arxiv.org/abs/2407.01449) approach and consists of the following components:

(1) Vision-Language model : granite-vision-3.3-2b (https://huggingface.co/ibm-granite/granite-vision-3.3-2b).

(2) Projection layer: linear layer that projects the hidden layer dimension of Vision-Language model to 128 and outputs 729 embedding vectors per image.

The scoring is computed using MaxSim-based late interaction mechanism.

**Training Data:**
Our training data is entirly comprised from DocFM. DocFM is a large-scale comprehensive dataset effort at IBM consisting of 85 million document pages extracted from unique PDF
documents sourced from Common Crawl, Wikipedia, and ESG (Environmental, Social, and Governance)
reports.

**Infrastructure:**
We train granite-vision-3.3-2b-embedding on IBM’s cognitive computing cluster, which is outfitted with NVIDIA A100 GPUs.

**Ethical Considerations and Limitations:**
The use of Large Vision and Language Models involves risks and ethical considerations people must be aware of, including but not limited to: bias and fairness, misinformation, and autonomous decision-making. Granite-vision-3.3-2b-embedding is not the exception in this regard. Although our alignment processes include safety considerations, the model may in some cases produce inaccurate or biased responses.
Regarding ethics, a latent risk associated with all Large Language Models is their malicious utilization. We urge the community to use granite-vision-3.3-2b-embedding with ethical intentions and in a responsible way.

**Resources**
- 📄 Granite Vision technical report [here](https://arxiv.org/abs/2502.09927)
- 📄 Real-MM-RAG-Bench paper (ACL 2025) [here](https://arxiv.org/abs/2502.12342)
- 📄 Vidore 2 paper [here](https://www.arxiv.org/pdf/2505.17166) 
- ⭐️ Learn about the latest updates with Granite: https://www.ibm.com/granite
- 🚀 Get started with tutorials, best practices, and prompt engineering advice: https://www.ibm.com/granite/docs/
- 💡 Learn about the latest Granite learning resources: https://ibm.biz/granite-learning-resources
