# Source-Relaxed Domain Adaptation

Pytorch implementation of our MICCAI 2020 paper entitled: Source-Relaxed Domain Adaptation for Image Segmentation. We compare to adversarial adaptation, using the method proposed by Tsai et al. here : https://arxiv.org/abs/2005.03697, as well as to Curriculum Domain Adaptation (Zhang et al.) https://arxiv.org/pdf/1812.09953.pdf.

# Paper

[Source-Relaxed Domain Adaptation for Image Segmentation.](https://arxiv.org/abs/2005.03697)
 <br />Mathilde Bateson, Jose Dolz, Hoel Kervadec, Herve Lombaert, Ismail Ben Ayed <br />
International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI).

Please cite our paper if you find it useful for your research.
```
@incollection{Bateson2020,
  doi = {10.1007/978-3-030-59710-8_48},
  url = {https://doi.org/10.1007/978-3-030-59710-8_48},
  year = {2020},
  publisher = {Springer International Publishing},
  pages = {490--499},
  author = {Mathilde Bateson and Hoel Kervadec and Jose Dolz and Herv{\'{e}} Lombaert and Ismail Ben Ayed},
  title = {Source-Relaxed Domain Adaptation for Image Segmentation},
  booktitle = {Medical Image Computing and Computer Assisted Intervention {\textendash} {MICCAI} 2020}
}
```

# Installation
Install PyTorch from http://pytorch.org with Python 3.6 and CUDA 8.0

# Dataset
The dataset is available at https://ivdm3seg.weebly.com/data.html. We used the Water (Wat) modality as the Source domain, and the In-Phase (IP) modality as the Target domain, and then reversed. Note: in our experiments we rotated the data from the sagittal plane to the transverse plane.

Download the data and put in the `data/sagittal` folder, then rotate and save into the `data/transverse` folder, both for the Wat and the IP modality:

```
python rotate.py  --base_folder='./data/transverse/IP/' --folders=['train','val'] --save_folder='./data/sagittal/IP/' --rot=’rot’  --grp_regex="Subj_\\d+_"
```

# Testing

Train the model with constrained domain adaptation. Create the Results folder `results/IP`. Results will be saved in the result folder `results/IP/Constraint/`
```
python main.py --dataset='data/transverse/' --target_dataset='data/transverse/' 
                         --folders="[('Wat', png_transform, False), ('GT', gt_transform, False)]"
                         --target_folders="[('IP', png_transform, False), ('GT', gt_transform, False)]+"
                                        "[('GT', gt_transform, False), ('GT', gt_transform, False)]"
                         --grp_regex='Subj_\\d+_\\d+'
                         --losses="[('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 0),"
                                " ('NaivePenalty', {'idc': [1]},'PreciseBounds', {'margin': 0.1, 'mode': 'percentage'}, 'soft_size', 2.5)]"
                         --losses_source="[('CrossEntropy',{'idc': [0,1],'weights':[1,1]},None,None,None, 1)]"
                          --model_weights='pretrained_source.pkl' --n_class=2 --in_memory=False, 
                          --network='ENet' --resize=36 --scheduler='DummyScheduler' 
                         scheduler_params='{}', workdir='results/Constraint'
```

