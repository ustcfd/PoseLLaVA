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
**We will organize the multimodal datasets for each pose sub-task according to the example below and follow the format for each part of the training dataset. In this format, `<image>` and `<pose>` serve as placeholders for two different input modalities: the image and the SMPL parameters.**

<details>
<summary>Example for Pose Generation task </summary>

```json
[
  {
    "id": "135000",
    "target_pose": ["Ground truth SMPL parameters.... "],
    "conversations": [
      {
        "from": "human",
        "value": "There is a person: torso is slightly leaning to the left, right arm is straight forward and hand is pointed down, left arm is down and bent up with the hand in front of chin Please output this person's SMPL pose."
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
</details>

<details>
<summary>Inputing image example for Pose Estimation or Pose Adjustment task</summary>

```json
[
  {
    "id": "135000",
    "image": "S8_WalkDog_1.55011271_000026.jpg",
    "target_pose": ["Ground truth SMPL parameters.... "],
    "conversations": [
      {
        "from": "human",
        "value": "<image> + pose estimation/adjustment instruction"
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
</details>

<details>
<summary>Inputing SMPL Parameters example for Pose Adjustment task </summary>

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
</details>

### Training Dataset Configuration

We configure the training datasets for all tasks using a single JSON file `ds_config/train_ds.json`. The `root` specifies the directory of the image files, while SMPLpose is left empty. The `annotation` represents the path to the training files, and `repeat_time` and `length` can be used together to configure the ratio of training data for each part.

### Finetune with LoRA

Set the parameters in the script `shell/posellava_sft_lora.sh`, where `meta_path` is the path to the training dataset configuration file. To run the training script, use the following command: 

```bash
bash shell/posellava_sft_lora.sh
```

## Evaluation

Similar to the training setup, configure the evaluation datasets for each sub-task in the `ds_config/eval_ds.json` JSON file. In the evaluation script, set the path to the fine-tuned model weights. To run the evaluation script, use the following command: 

```bash
bash shell/run_inference.sh
```

## Streamlit demo
We provide a Streamlit-based web demo that loads the trained model, offering a visual demonstration of its performance. Note that this demo does not support SMPL parameter input, as entering 72 SMPL parameters through the frontend is impractical. However, our PoseLLaVA does support this, and you can test the SMPL input effect using the evaluation script.

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
- [lmms-finetune](https://github.com/zjysteven/lmms-finetune): a minimal codebase for finetuning large multimodal models.
- [PoseGPT](https://github.com/yfeng95/PoseGPT): a Multi-modal LLM to understand and reason about 3D Human poses.
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): an amazing open-source project of LMM.