# PoseLLaVA: Pose Centric Multimodal LLM for Fine-Grained 3D Pose Manipulation

<div align=center>

[![arXiv preprint](http://img.shields.io/badge/arXiv-2412.11634-b31b1b)](https://arxiv.org/abs/2412.11634)
[![Code](https://img.shields.io/badge/Code-HDR-yellow)](https://github.com/yeungchenwa/HDR)
</div>

<video width="600" controls>
  <source src="papers/posellava_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Getting Started

## 🔥 Model Zoo
| **Model**     | **chekcpoint**                                            | **status** |
|---------------|-----------------------------------------------------------|------------|
| **PoseLLava** | [Huggingface](https://huggingface.co/FengD2023/Posellava) | Released  |

## 🔥 Dataset Zoo
| **Model**                                    | **chekcpoint** | **status** |
|----------------------------------------------|----------------|------------|
| **PosePart**                              | [GoogleDrive](https://drive.google.com/drive/folders/1K-k_3LaPaXlTjUWpOvlehAC6Yxqiv9RP?usp=drive_link) | Released  |

## Training
### Dataset Preparation


<details>
<summary>Image Input example for Pose Estimation </summary>

```json
[
  {
    "id": "135000",
    "image": "S8_WalkDog_1.55011271_000026.jpg",
    "target_pose": ["Ground truth SMPL parameters.... "],
    "conversations": [
      {
        "from": "human",
        "value": "<image> Can you predict the SMPL pose of the person in this image?"
      },
      {
        "from": "gpt",
        "value": "Sure, it is <POSE>."
      }
    ]
  }
  ...
]
```

<details>
<summary>SMPL Pose Parameters example for Pose Adjustment task </summary>

```json
[
  {
    "id": "135000",
    "input_pose": ["Initialized SMPL parameters.... "],
    "target_pose": ["Ground truth SMPL parameters.... "],
    "conversations": [
      {
        "from": "human",
        "value": "<pose> Please peruse the description below. Extend your right arm to the right and back while keeping your right upper arm flat, with straight knees and feet shoulder-width apart. Start with the pose and use the textual description to generate the corresponding adjusted SMPL pose."
      },
      {
        "from": "gpt",
        "value": "The SMPL pose of the person is <POSE>."
      }
    ]
  }
  ...
]
```

### Finetune with LoRA
```bash
bash shell/posellava_sft_lora.sh
```

### Inference
```bash
bash shell/run_inference.sh
```

### Streamlit demo
```bash
bash shell/posellava_demo.sh
```

## Citation
```bibtex
@inproceedings{feng2025posellava,
  title={PoseLLaVA: Pose Centric Multimodal LLM for Fine-Grained 3D Pose Manipulation},
  author={Feng, Dong and Guo, Ping and Peng, Encheng and Zhu, Mingmin and Yu, Wenhao and Wang, Peng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2951--2959},
  year={2025}
}
```
## Acknowledgments 

The project is based on 
- [lmms-finetune](https://github.com/zjysteven/lmms-finetune): A minimal codebase for finetuning large multimodal models.
- [PoseGPT](https://github.com/yfeng95/PoseGPT): a Multi-modal LLM to understand and reason about 3D Human poses.
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): An amazing open-source project of LMM.