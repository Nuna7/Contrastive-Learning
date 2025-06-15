## 📊 Evaluation of SoM-LLava and PixelCLIP on Multi-View Images

### 📁 Directory Structure

- **Evaluation root path for SoM-LLava**:  
  `/SoM/`

- **Evaluation root path for PixelCLIP**:  
  `/PixelCLIP/`

- **Input images**:  
  `/SoM/images/`
  `/PixelCLIP/images/`

- **Output results**:  
  `/SoM/outputs/`
  `/PixelCLIP/outputs/`

### 🧠 Models Used

#### 🔍 Vision Models
- SemanticSAM (`semsam`)
- Segment Everything Everywhere All At Once (`seem`)
- PixelCLIP

#### 💬 Language Models
- **LLaVA 1.5 (13B)**  
  - Referred to as `huggingface` in output directory  
  - Trained specifically on the set of masks
- **LLaVA 1.6 (13B)**  
  - Referred to as `ollama` in output directory

> In `/SoM/outputs/`:
> - `huggingface/` → Results from LLaVA 1.5 (13B)  
> - `ollama/` → Results from LLaVA 1.6 (13B)  
> - `prompt1/` and `prompt2/` → Represent different final outputs based on prompt variations

---

### 🧪 Evaluation Code

- Evaluation code is located at:  
- **SOM-Lllava**:
  `/SoM/set_off_mask_model.py`

- **PixelCLIP**:
  `/PixelCLIP/main.py`

---

### 📦 Checkpoint Setup

1. Create a directory at:  
   `/SoM/checkpoints/`
   `/PixelCLIP/weights/`

2. Navigate to the script:  
- **SOM-Lllava**:
   `/SoM/download_ckpt.sh`

   Run the **first two commands** in the script to download checkpoints for:
   - `seem`
   - `semsam`

- **PixelCLIP**:
    `wget https://huggingface.co/hsshin98/PixelCLIP/resolve/main/weights/pixelclip_vit_base.pth -O PixelCLIP/weights/pixelclip_vit_base_sa1b/model_final.pth`

### Environment
Both models use different conda environment with their corresponding yml file `som_environments.yml` and `pixelclip_environments.yml`

### Note
Clone the repository as submodule.

```bash
git clone --recurse-submodules https://github.com/Nuna7/Contrastive-Learning.git
git submodule update --init --recursive
```