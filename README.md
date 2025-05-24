# 3D CAVLA: Leveraging Depth and 3D Context to Generalize Vision Language Action Models for Unseen Tasks

<p align="center"><strong><em>Accepted at the 3D LLM/VLA Workshop @ CVPR 2025</em></strong></p>

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

# Additional dependencies for LIBERO experiments
pip install bddl easydict cloudpickle gym
pip install robosuite==1.4.0
pip install imageio[ffmpeg]
```

---

## LIBERO Configuration File

Before running training or testing, update the LIBERO config file. After cloning this repo, you can find the file at:

```
../.libero/config.yaml
```

Edit the file so it looks like this (replace `<path_to_local_3dcavla>` with your local path):

```yaml
assets: <path_to_local_3dcavla>/LIBERO/libero/libero/./assets
bddl_files: <path_to_local_3dcavla>/LIBERO/libero/libero/./bddl_files
benchmark_root: <path_to_local_3dcavla>/LIBERO/libero/libero
datasets: <path_to_local_3dcavla>LIBERO/libero/libero/../datasets
init_states: <path_to_local_3dcavla>LIBERO/libero/libero/./init_files
```

---

## Dataset Download (~293 GB)

We use a modified version of the LIBERO dataset with depth maps. You can download it from Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/datasets/bhatvineet/modified_libero_rlds_cotdep
```

---

## Fine-Tuning

Fine-tuning is done using the `train.sh` script. Adjust the number of GPUs and paths as needed:

```bash
bash train.sh
```

---

## Evaluation

To evaluate a trained model, use the `test.sh` script:

```bash
bash test.sh
```

---

## Citation

If you use this codebase, please cite:

```bibtex
@article{bhat20253d,
  title={3D CAVLA: Leveraging Depth and 3D Context to Generalize Vision Language Action Models for Unseen Tasks},
  author={Bhat, Vineet and Lan, Yu-Hsiang and Krishnamurthy, Prashanth and Karri, Ramesh and Khorrami, Farshad},
  journal={arXiv preprint arXiv:2505.05800},
  year={2025}
}
```

---

## 🧑‍💻 Acknowledgements

Parts of this codebase build on [OpenVLA-OFT](https://github.com/moojink/openvla-oft) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). The `LIBERO/` folder is included in this repo with minor environment modifications for compatibility and reproducibility.

---




