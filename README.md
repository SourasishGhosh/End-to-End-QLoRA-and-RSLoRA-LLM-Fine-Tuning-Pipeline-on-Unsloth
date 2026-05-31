

# 🦥 Large Language Model Parameter-Efficient Fine-Tuning (PEFT) Pipeline

This repository hosts an end-to-end, modular Parameter-Efficient Fine-Tuning (PEFT) framework powered by **Unsloth** and **Hugging Face Transformers**. The pipeline is explicitly architecture-optimized to execute low-rank quantization adaptations on compute-restricted consumer hardware (tested on an **NVIDIA GeForce RTX series 4GB Laptop GPU** under Windows).

The project processes unstructured document resources into instruction-tuned synthetic conversational datasets, runs comparative configuration experiments across separate PEFT architectures, tracks performance scalars using TensorBoard metrics, and provides a stable runtime generation engine.

---

##  System Architecture & Workspace Matrix

To bypass compilation blocks, Windows system locks, and broken Triton dependency dependencies, the environment is strictly locked to the following execution matrix:

* **Core Tensor Engine:** PyTorch `2.5.1+cu121`

* **Compute Driver Backend:** NVIDIA CUDA Toolkit `12.1`

* **Acceleration Framework:** Unsloth & Unsloth Zoo (Eager Evaluation Runtime Engine)

* **Target Foundation Base Model:** `unsloth/gemma-2-2b-it` (Instruction Tuned)

### Project Directory Layout

```text

lora-qlora/
├── .venv/                     
├── .gitignore                 
├── config.py                  
├── data.py                    
├── dataset.jsonl              # Synthetic instruction-tuned instruction training pairs
├── inference.py               # Evaluator inference generation execution script
├── main.py                    
├── make_dataset.py            # Automated document batching API compilation engine
├── train.py                   # SFTTrainer training loop and logging callback mapping
└── README.md        

```
------------------------------------

### **Sequential Execution Instructions**



Ensure your virtual environment is initialized and activated inside your PowerShell terminal before triggering the processing steps:

PowerShell

```
.venv\Scripts\activate

```

### Step 1: Synthetic Training Set Compilation

Extract knowledge context maps from raw PDF documents inside `dataset/pdfs/`. This processes text data through an optimized batch-context array layout to generate high-quality data entries while preserving API rate restrictions:

PowerShell

```
python make_dataset.py

```

*Output Target:* `dataset/dataset.jsonl` containing structured chat message turns.

### Step 2: Run Quantized Low-Rank Adaptation (QLoRA) Training

Launch fine-tuning using compressed 4-bit NormalFloat4 (NF4) base parameters layered with active training weight matrices. Telemetry logging outputs are captured dynamically step-by-step:

PowerShell

```
python main.py --fine_tuning qlora --output_dir ./gemma_qlora

```

### Step 3: Run Rank-Stabilized LoRA (rsLoRA) Training

Launch an experimental companion fine-tuning run using a dynamically scaling dimension modifier ($\Delta W \propto \frac{\alpha}{\sqrt{r}}$) to stabilize backpropagation gradients:

PowerShell

```
python main.py --fine_tuning rslora --output_dir ./gemma_rslora

```

### Step 4: Execute Native Inference Generation

Verify vocabulary predictions and generation performance directly on your specialized adapters. The runtime engine manually injects a balanced attention mask layer to ensure consistent text formatting outputs:

PowerShell

```
python inference.py

```
----------------------------------
### **Comparative Analytics Dashboard**



The pipeline automatically logs metrics including training loss adjustments, learning rate schedules, and gradient norms into local event files via a TensorBoard callback handler.

### Launch Dashboard Tracker

Point your local analytics tracker directly to your project's parent directory to map and compare the metrics profiles of both training configurations simultaneously:

PowerShell

```

tensorboard --logdir="D:\PERSONAL PROJECTS\lora-qlora"

```

### Graphical Insights

Open your web browser and navigate to `http://localhost:6006/` to inspect your plots.

1\.  **`train/loss`:** Maps overall model error reduction. Highlights the point where the models successfully resolved data structural noise and began adjusting weights cleanly.

2\.  **`train/grad_norm`:** Displays the internal scaling behaviors of your parameters. Highlights the structural contrast between the low, conservative updates of QLoRA vs. the assertive learning momentum calculated by rsLoRA.

3\.  **`train/learning_rate`:** Tracks your optimization step size changes, confirming that your linear decay schedule applied evenly across both training configurations.

------

* For further info contact:   `sourasishghosh02@gmail.com`