## 📊 Evaluation of SoM-LLava on Multi-View Images

### 📁 Directory Structure

- **Evaluation root path**:  
  `/SoM/`

- **Input images**:  
  `/SoM/images/`

- **Output results**:  
  `/SoM/outputs/`

### 🧠 Models Used

#### 🔍 Vision Models
- SemanticSAM (`semsam`)
- Segment Everything Everywhere All At Once (`seem`)

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
  `/SoM/set_off_mask_model/py`

---

### 📦 Checkpoint Setup

1. Create a directory at:  
   `/SoM/checkpoints/`

2. Navigate to the script:  
   `/SoM/download_ckpt.sh`

3. Run the **first two commands** in the script to download checkpoints for:
   - `seem`
   - `semsam`
