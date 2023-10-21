# IoU-based positives for CT

PyTorch implementation and results for *IoU-based positives*, a novel strategy
for generating positive image crops for visual self-supervised learning.

The method can replace standard random cropping for self-supervised methods such as
SimCLR, DINO and iBOT. When compared to plain random cropping on pretraining with
2D CT medical images, *IoU-based positives* **yields better results where fine-tuning on a downstream organ segmentation task**.

### Why alternative cropping strategies?

It's been noticed in a few recent papers (see [References](#references)) that 
**plain random cropping for generating positive pairs might be suboptimal due to potential significant sementical misalignment**. 
While the misalignment might not be well visible on images from ImageNet, it's much clearer when considering medical images:

![ImageNet vs CT](.github/imagenet_vs_ct.png?raw=true)

Contrary to previous works presenting alternative strategies for generating positive crops for medical data (see [References](#references)), *IoU-based positives* **does not require any metadata generalizes to both 2D and 3D data, uses no external models and works seamlessly with heterogeneous datasets.**

### Details

The proposed method works as follows. Given a 2D or 3D input image x,
sample 2 positive crops x1 , x2 so that IoU (x1 , x2 ) ∈ [imin , imax ], where imin and imax cor-
respond to minimum and maximum IoU allowed for the crops to be considered positive. The
intuition behind such an approach is that for some of the domains, semantically consistent crops
appear mostly within a single area, while crops from remote areas might depict various down-
stream classes. On the other hand, too high overlap might introduce easy positives that don’t
create a training signal.

### Results

The values of imin and imax must be determined empirically. To
assess optimal imin and imax for medical imaging pretraining, a few non-overlapping IoU in-
tervals were evaluated by first pretraining with a given interval using SimSiam, then fine-tuning
on organ segmentation task. 2 pretrainings were run per IoU interval and 2 fine-tunings were run per
pretrained model, which resulted in 4 DSC scores per IoU interval. The non-
overlapping intervals were chosen so that they would reflect various levels of semantic and
visual similarity between the positive crops. Plain random cropping was also evaluated.

| IoU interval    |Pretraining 1|Pretraining 2| Mean DSC                       | DSC 1   | DSC 2   | DSC 3   | DSC 4   |
|:----------------|:-----------:|:-----------:|:------------------------------:|:-------:|:-------:|:-------:|:-------:|
| -               |-            |-            |$0.843 \pm 0.004$              | [$0.842$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/3bx73fx8) | [$0.848$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/kbt60o1y) | [$0.846$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/uy6p9aoj) | [$0.836$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/7jj8xaxp) |
| $[0, 0]$        |[link](https://wandb.ai/dittohead/exploring-ssl-for-ct-pre/runs/jr49k3cg)|[link](https://wandb.ai/dittohead/exploring-ssl-for-ct-pre/runs/lske2j87)|$0.838 \pm 0.007$              | [$0.836$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/i5h0zcm9) | [$0.831$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/xf5vz6wj) | [$0.834$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/z9ryr6qx) | [$0.850$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/bx0qr73r) |
| $[0.0001, 0.3]$ |[link](https://wandb.ai/dittohead/exploring-ssl-for-ct-pre/runs/i5clmwif)|[link](https://wandb.ai/dittohead/exploring-ssl-for-ct-pre/runs/6t60ynal)|$0.839 \pm 0.012$              | [$0.847$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/unsew1xa) | [$0.831$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/vnpkqlwm) | [$0.855$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/mt03b6xe) | [$0.824$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/cxdy86mw) |
| $[0.3, 0.6]$    |[link](https://wandb.ai/dittohead/exploring-ssl-for-ct-pre/runs/7wrcwqc9)|[link](https://wandb.ai/dittohead/exploring-ssl-for-ct-pre/runs/86bsy8uu)|$\boldsymbol{0.853 \pm 0.002}$ | [$0.854$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/ejghu868) | [$0.851$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/xcb50xmp) | [$0.851$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/i8278cxy) | [$0.857$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/42gn2rby) |
| $[0.6, 1]$      |[link](https://wandb.ai/dittohead/exploring-ssl-for-ct-pre/runs/vpay7gn6)|[link](https://wandb.ai/dittohead/exploring-ssl-for-ct-pre/runs/kk0rhojm)|$0.843 \pm 0.013$              | [$0.850$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/dtdwsaw8) | [$0.855$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/wwq8czji) | [$0.848$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/202owflv) | [$0.822$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/42gn2rby) |
| Random crop.    |[link](https://wandb.ai/dittohead/exploring-ssl-for-ct-pre/runs/lkjkduxz)|[link](https://wandb.ai/dittohead/exploring-ssl-for-ct-pre/runs/4x7738lr)|$0.824 \pm 0.011$              | [$0.815$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/qutw4bmq) | [$0.841$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/tbc0d1no) | [$0.828$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/y29bwpxv) | [$0.812$](https://wandb.ai/dittohead/exploring-ssl-for-ct-tune/runs/afgrghq6) |


### Data

[FLARE 2022 challenge](https://flare22.grand-challenge.org/) data was used both for pretraining and fine-tuning. The training set includes 50 CT scans with
voxel-level labels of 13 abdominal organs and 2,000 unlabeled CT scans. The validation set
includes 50 visible unlabeled cases. The testing set includes 200 hidden cases.
The present work utilized the challenge’s training set only. For pretraining of the backbone,
all the 2,000 unlabeled CT scans were used without any metadata. For fine-tuning, 50 labeled cases were utilized. To evaluate the IoU intervals thoroughly, a majority of
cases were assigned to a validation subset: 35 cases were included in the fine-tuning validation
subset and 15 cases were included in the fine-tuning training subset.

2D CT slices were used for pretraining and fine-tuning instead of 3D due to
computation costs.

### Requirements

See [requirements.txt](./requirements.txt).

### How to run

1. Preprocess CTs using `preprocess_flare_labelled.py` and
`preprocess_flare_unlabelled.py` (default args values were used for the experiments).
This is to extract 2D .png from 3D .nii.gz files + there's no need to repeat
the same processing each time image is loaded during training.

2. Run pretraining(s).

```console
python main_simsiam.py --data_dir <DIR_WITH_PREPROCESSED_2D_PNGS> --embedding_size 48 --batch_size 128 --n_epochs 100 --base_lr 0.025 --min_iou <I_MIN> --max_iou <I_MAX> --num_workers <NUM_WORKERS> --use_amp
```

3. Run fine-tuning(s).

```console
python main_finetune.py --data_dir $PLG_GROUPS_STORAGE/plggsslct/finetune_preprocessed_2d --chkpt_path <PATH_TO_PRETRAINED_CHKPT> --embedding_size 48 --batch_size 32 --n_epochs 225 --patience 20 --sw_batch_size 64 --ignore_user_warning --num_workers <NUM_WORKERS> --use_amp
```

Arguments used for running each experiment can be found in the corresponding
wandb runs (see the table in [Results][#results]): Files -> config.yaml.


### DINO and 3D data

There's an analogous script `main_dino.py` for pretraining with DINO. 
One can also run all the scripts for 3D data (using `--spatial_dims 3`). 
Be careful though! At the moment, I can't guarantee that it will work and your PC might blow
up. ;)


### References

[1] 
Senthil Purushwalkam and Abhinav Gupta. “Demystifying Contrastive Self-Supervised Learning: Invariances, Augmentations and Dataset Biases”. Advances in Neural Information Processing Systems, 2020.

[2]
Xiangyu Peng et al. “Crafting Better Contrastive Views for Siamese Representation Learning”. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[3]
Shekoofeh Azizi et al. “Big Self-Supervised Models Advance Medical Image Classification”. IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[4]
Yen Nhi Truong Vu et al. “MedAug: Contrastive learning leveraging patient metadata improves representations for chest X-ray interpretation”. Proceedings of the 6th Machine Learning for Healthcare Conference, 2021.

[5]
Dewen Zeng et al. “Positional Contrastive Learning for Volumetric Medical Image Segmentation”. Medical Image Computing and Computer Assisted Intervention – MIC-
CAI, 2021.

[6]
Yankai Jiang et al. Anatomical Invariance Modeling and Semantic Alignment for Self-supervised Learning in 3D Medical Image Segmentation. arXiv: 2302.05615
[cs.CV], 2023.