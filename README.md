# FEAT
This is the code of CVPR 2020 oral paper "Distilling Cross-Task Knowledge via Relationship Matching". If you use any content of this repo for your work, please cite the following bib entry:

    @inproceedings{ye2020refilled,
      author    = {Han-Jia Ye and
                   Su Lu and
                   De-Chuan Zhan},
      title     = {Cross-Task Knowledge Distillation via Relationship Matching},
      booktitle = {Computer Vision and Pattern Recognition (CVPR)},
      year      = {2020}
    }
    
## Cross-Task Knowledge Distillation
It is intuitive to take advantage of the learning experience from related pre-trained models to facilitate model training in the current task. Different from ﬁne-tuning or parameter regularization, knowledge distillation/knowledge reuse extracts kinds of dark knowledge/privileged information from a ﬁxed strong model (a.k.a. "teacher"), and enrich the target model (a.k.a. "student") training with more signals. Owing to the strong correspondence between classiﬁer and class,it is difﬁcult to reuse the classiﬁcation knowledge from a cross-task teacher model.

<img src='figures/setting.png'>
