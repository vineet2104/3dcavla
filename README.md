# 3D CAVLA: Leveraging Depth and 3D Context to Generalize Vision Language Action Models for Unseen Tasks

🔗 **Project Website**: https://3d-cavla.github.io

📄 **Paper**: https://arxiv.org/abs/2505.05800v1

## 🧠 Abstract

Robotic manipulation in 3D requires learning an Ndegree-
of-freedom joint space trajectory of a robot manipulator.
Robots must possess semantic and visual perception abil-
ities to transform real-world mappings of their workspace
into the low-level control necessary for object manipula-
tion. Recent work has demonstrated the capabilities of
fine-tuning large Vision-Language Models (VLMs) to learn
the mapping between RGB images, language instructions,
and joint space control. These models typically take as in-
put RGB images of the workspace and language instruc-
tions, and are trained on large datasets of teleoperated
robot demonstrations. In this work, we explore methods
to improve the scene context awareness of a popular re-
cent Vision-Language-Action model by integrating chain-
of-thought reasoning, depth perception, and task-oriented
region of interest detection. Our experiments in the LIBERO
simulation environment show that our proposed model, 3D-
CAVLA, improves the success rate across various LIBERO
task suites, achieving an average success rate of 98.1%.
We also evaluate the zero-shot capabilities of our method,
demonstrating that 3D scene awareness leads to robust
learning and adaptation for completely unseen tasks. 3D-
CAVLA achieves an absolute improvement of 8.8% on un-
seen tasks.

---

## 🛠 Environment Installation

Follow these steps to set up the `cavla3d` environment:

```bash
# Create and activate conda environment
conda create -n cavla3d python=3.10 -y
conda activate cavla3d

# Install PyTorch
# Choose the correct command from https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone the repository and install dependencies
git clone https://github.com/vineet2104/3dcavla.git
cd cavla3d
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja
ninja --version; echo $?  # Should return 0 if installed correctly
pip install "flash-attn==2.5.5" --no-build-isolation
```

---

## 📦 Dataset Download

We use a modified version of the LIBERO dataset adapted for causal action-aware training. You can download it from Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/datasets/bhatvineet/modified_libero_rlds_cotdep
```

---

## 🚀 Fine-Tuning

Fine-tuning is done using the `train.sh` script. Adjust the number of GPUs and paths as needed:

```bash
bash train.sh
```

---

## 🥪 Evaluation

To evaluate a trained model, use the `test.sh` script:

```bash
bash test.sh
```

---

## 📚 Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{
  bhat2025dcavla,
  title={3D-{CAVLA}: Leveraging Depth and 3D Context to Generalize Vision{\textendash}Language Action Models for Unseen Tasks},
  author={Vineet Bhat and Yu-Hsiang Lan and Prashanth Krishnamurthy and Ramesh Karri and Farshad Khorrami},
  booktitle={CVPR 2025 Workshop on 3D-LLM/VLA: Bridging Language, Vision and Action in 3D Environments},
  year={2025},
  url={https://openreview.net/forum?id=hn2VHDr95j}
}
```

---

## 🧑‍💻 Acknowledgements

Parts of this codebase build on [OpenVLA-OFT](https://github.com/moojink/openvla-oft) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). The `LIBERO/` folder is included in this repo with minor environment modifications for compatibility and reproducibility.

---
