---
base_model:
- ibm-granite/granite-4.0-tiny-preview
license: apache-2.0
pipeline_tag: text-generation
tags:
- language
- granite-4.0
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
<li>Original Model: <a href="https://huggingface.co/ibm-granite/granite-4.0-tiny-preview">ibm-granite/granite-4.0-tiny-preview</a></li>
</ul>
</div>

<br>

# Provided Files

Get started on your IBM Z & LinuxONE system

```sh
build/bin/llama-server -hf taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF:Q4_K_M
```

| Name | Quant Method | Bits | Size | Use Case |
|---|---|---|---|---|
| [Granite-4.0-Tiny-Preview-BE-Q2_K.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q2_K.gguf) | Q2_K | 2 | 2.3G | smallest K-quant, significant quality loss |
| [Granite-4.0-Tiny-Preview-BE-Q3_K_L.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q3_K_L.gguf) | Q3_K_L | 3 | 3.2G | very small, high quality loss |
| [Granite-4.0-Tiny-Preview-BE-Q3_K_M.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q3_K_M.gguf) | Q3_K_M | 3 | 3.0G | very small, high quality loss |
| [Granite-4.0-Tiny-Preview-BE-Q3_K_S.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q3_K_S.gguf) | Q3_K_S | 3 | 2.7G | very small, high quality loss |
| [Granite-4.0-Tiny-Preview-BE-IQ4_NL.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-IQ4_NL.gguf) | IQ4_NL | 4 | 3.6G | small, good quality for its size |
| [Granite-4.0-Tiny-Preview-BE-IQ4_XS.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-IQ4_XS.gguf) | IQ4_XS | 4 | 3.4G | small, good quality for its size |
| [Granite-4.0-Tiny-Preview-BE-MXFP4_MOE.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-MXFP4_MOE.gguf) | MXFP4_MOE | 4 | 3.7G | MoE experts in MXFP4, other tensors Q8_0 |
| [Granite-4.0-Tiny-Preview-BE-Q4_0.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q4_0.gguf) | Q4_0 | 4 | 3.5G | legacy format, prefer Q4_K_M |
| [Granite-4.0-Tiny-Preview-BE-Q4_1.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q4_1.gguf) | Q4_1 | 4 | 3.9G | legacy format, prefer Q4_K_M |
| [Granite-4.0-Tiny-Preview-BE-Q4_K_M.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q4_K_M.gguf) | Q4_K_M | 4 | 3.8G | medium, balanced quality - recommended |
| [Granite-4.0-Tiny-Preview-BE-Q4_K_S.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q4_K_S.gguf) | Q4_K_S | 4 | 3.6G | medium, balanced quality - recommended |
| [Granite-4.0-Tiny-Preview-BE-Q5_0.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q5_0.gguf) | Q5_0 | 5 | 4.3G | legacy format, prefer Q5_K_M |
| [Granite-4.0-Tiny-Preview-BE-Q5_1.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q5_1.gguf) | Q5_1 | 5 | 4.7G | legacy format, prefer Q5_K_M |
| [Granite-4.0-Tiny-Preview-BE-Q5_K_M.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q5_K_M.gguf) | Q5_K_M | 5 | 4.4G | large, very low quality loss - recommended |
| [Granite-4.0-Tiny-Preview-BE-Q5_K_S.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q5_K_S.gguf) | Q5_K_S | 5 | 4.3G | large, very low quality loss - recommended |
| [Granite-4.0-Tiny-Preview-BE-Q6_K.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q6_K.gguf) | Q6_K | 6 | 5.1G | very large, extremely low quality loss |
| [Granite-4.0-Tiny-Preview-BE-Q8_0.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-Q8_0.gguf) | Q8_0 | 8 | 6.6G | very large, extremely low quality loss |
| [Granite-4.0-Tiny-Preview-BE-BF16.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-BF16.gguf) | BF16 | 16 | 12.4G | half precision, effectively lossless |
| [Granite-4.0-Tiny-Preview-BE-F16.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-F16.gguf) | F16 | 16 | 12.4G | half precision, effectively lossless |
| [Granite-4.0-Tiny-Preview-BE-F32.gguf](https://huggingface.co/taronaeo/Granite-4.0-Tiny-Preview-BE-GGUF/blob/main/Granite-4.0-Tiny-Preview-BE-F32.gguf) | F32 | 32 | 24.9G | full precision, largest |

---

[![mof-class3-qualified](https://mot.isitopen.ai/modules/mof/assets/badge_class3_qualified.png)](https://mot.isitopen.ai/model/1168)

# Granite-4.0-Tiny-Preview

**Model Summary:**
Granite-4-Tiny-Preview is a 7B parameter fine-grained hybrid mixture-of-experts (MoE) instruct model fine-tuned from Granite-4.0-Tiny-Base-Preview using a combination of open source instruction datasets with permissive license and internally collected synthetic datasets tailored for solving long context problems. This model is developed using a diverse set of techniques with a structured chat format, including supervised fine-tuning, and model alignment using reinforcement learning.

- **Developers:** Granite Team, IBM
- **Website**: [Granite Docs](https://www.ibm.com/granite/docs/)
- **Release Date**: May 2nd, 2025
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

**Supported Languages:** 
English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese. However, users may fine-tune this Granite model for languages beyond these 12 languages.

**Intended Use:** 
This model is designed to handle general instruction-following tasks and can be integrated into AI assistants across various domains, including business applications.

**Capabilities**
* Thinking
* Summarization
* Text classification
* Text extraction
* Question-answering
* Retrieval Augmented Generation (RAG)
* Code related tasks
* Function-calling tasks
* Multilingual dialog use cases
* Long-context tasks including long document/meeting summarization, long document QA, etc.

**Installation:** 
You need to install transformer from source to use this checkpoint.
<!-- This is a simple example of how to use Granite-4.0-Tiny-Base-Preview model. -->

<!-- Usage: Install transformer from source or use transformer version v4.45 to use this checkpoint. -->

HuggingFace PR: https://github.com/huggingface/transformers/pull/37658

Install transformer from source: https://huggingface.co/docs/transformers/en/installation#install-from-source
<!-- While the native support of this model in Hugging Face Transformers is pending ([PR](https://github.com/huggingface/transformers/pull/37658)), you need to install transformers from the following source to use this model:
```shell
git clone https://github.com/Ssukriti/transformers.git
cd transformers
git checkout granitemoe_hybrid_external_cleanup
pip install -e .
``` -->
<!-- Install the following libraries:

```shell
pip install torch torchvision torchaudio
pip install accelerate
pip install transformers
``` -->
**Generation:**
After installation, copy the code snippet below to run the example.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

model_path="ibm-granite/granite-4.0-tiny-preview"
device="cuda"
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
tokenizer = AutoTokenizer.from_pretrained(
        model_path
)

conv = [{"role": "user", "content":"You have 10 liters of a 30% acid solution. How many liters of a 70% acid solution must be added to achieve a 50% acid mixture?"}]

input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", thinking=True, return_dict=True, add_generation_prompt=True).to(device)

set_seed(42)
output = model.generate(
    **input_ids,
    max_new_tokens=8192,
)

prediction = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
print(prediction)
```

**Evaluation Results:**

<table>
<thead>
    <caption style="text-align:center"><b>Comparison with previous granite models<sup id="fnref1"><a href="#fn1">1</a></sup>. Scores of AlpacaEval-2.0 and Arena-Hard are calculated with thinking=True</b></caption>
  <tr>
    <th style="text-align:left; background-color: #001d6c; color: white;">Models</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">Arena-Hard</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">AlpacaEval-2.0</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">MMLU</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">PopQA</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">TruthfulQA</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">BigBenchHard</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">DROP</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">GSM8K</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">HumanEval</th>
   <th style="text-align:center; background-color: #001d6c; color: white;">HumanEval+</th>
  <th style="text-align:center; background-color: #001d6c; color: white;">IFEval</th>
  <th style="text-align:center; background-color: #001d6c; color: white;">AttaQ</th>
  </tr></thead>
<tbody>
 
  <tr>
      <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;"><b>Granite-3.3-2B-Instruct</b></td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 28.86 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 43.45 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 55.88 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 18.4 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 58.97 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 52.51 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 35.98 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 72.48 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 80.51 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 75.68 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 65.8 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">87.47</td>
      </tr>
  <tr>
      <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Granite-3.3-8B-Instruct</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 57.56 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 62.68 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 65.54 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 26.17 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 66.86 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 59.01 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 41.53 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 80.89 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 89.73 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 86.09 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 74.82 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">88.5</td>
      </tr>
  <tr>
      <td style="text-align:left; background-color: #DAE8FF; color: black;"><b>Granite-4.0-Tiny-Preview</b></td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 26.70 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 35.16 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 60.40 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 22.93 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 58.07 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 55.71 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 46.22 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 70.05  </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 82.41 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 78.33 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 63.03 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 86.10 </td>
      </tr> 
</tbody></table>

**Training Data:** 
Overall, our training data is largely comprised of two key sources: (1) publicly available datasets with permissive license, (2) internal synthetically generated data targeted to enhance reasoning capabilities. 


**Infrastructure:**
We train Granite-4.0-Tiny-Preview using IBM's super computing cluster, Blue Vela, which is outfitted with NVIDIA H100 GPUs. This cluster provides a scalable and efficient infrastructure for training our models over thousands of GPUs.

**Ethical Considerations and Limitations:** 
Granite-4.0-Tiny-Preview, leverages both permissively licensed open-source and select proprietary data for enhanced performance. Since it inherits its foundation from the previous model, all ethical considerations and limitations applicable to [Granite-4.0-Tiny-Preview](https://huggingface.co/ibm-granite/granite-4.0-tiny-preview) remain relevant.

**Signature verification:**
Model signing is an experimental feature with ongoing development, which might include breaking changes. We are releasing these capabilities to improve the integrity of our models for our security-conscious users and to facilitate feedback from the community.

Before trying to verify the signature, ensure that the tensor files have been downloaded with git-lfs and that no files have been added, removed, or modified in your local git checkout:

```bash
  git lfs fetch --all
  git lfs pull
  git lfs checkout
```

Install the model_signing (v1.0.1) library with the following command:

```bash
   pip install 'model-signing==v1.1.1'
```



Then verify the signature with the following command ensuring that the IBM identity 'granite.preview@ibm.com' was used for signing this model:

```bash
   python -m model_signing verify sigstore \
     --signature model.sig \
     --ignore-paths .git \
     --ignore-paths .gitattributes \
     --identity Granite.Preview@ibm.com \
     --identity_provider https://sigstore.verify.ibm.com/oauth2 \
     .
```

**Resources**
- ⭐️ Learn about the latest updates with Granite: https://www.ibm.com/granite
- 📄 Get started with tutorials, best practices, and prompt engineering advice: https://www.ibm.com/granite/docs/
- 💡 Learn about the latest Granite learning resources: https://ibm.biz/granite-learning-resources
