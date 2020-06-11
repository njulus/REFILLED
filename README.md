# REFILLED
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

<img src='figures/setting.png' width='520' div align=center>

## Two-Stage Solution - REFILLED
We propose the RElationship FacIlitated Local cLassifiEr Distillation (REFILLED), which decomposes the knowledge distillation ﬂow for embedding and the top-layer classiﬁer respectively. REFILLED contains two stages. First, the discriminative ability of features is emphasized. For those hard triplets determined by the embedding of the student model, the teacher’s comparison between them is used as the soft supervision. A teacher enhances the discriminative embedding of the student by specifying the proportion for each object how much a dissimilar impostor should be far away from a target nearest neighbor. Furthermore, the teacher constructs the soft supervision for each instance by measuring its similarity to a local center. By matching the "instance-label" predictions across models, the cross-task teacher improves the learning efﬁcacy of the student.

<img src='figures/two_stage.png' width='800' div align=center>

## Experiment Results

## Code and Arguments
