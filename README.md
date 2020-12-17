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
REFILLED can be used in several applications, e.g., standard knowledge distillation, cross-task knowledge distillation and middle-shot learning. Standard knowledge distillation is widely used and we show the results under this setting below. Experiment results of cross-task knowledge distillation and middle-shot learning can be found in the paper.

**Dataset:CIFAR100**

**Teacher:wide_resnet-(40-2)**

**Student:wide_resnet-{(40,2),(16,2),(40,1),(16,1)}**
|Dataset: CIFAR-100|
|Teacher: wide_resnet-(40,2)|
|Student: wide_resnet-{(40,2), (16,2), (40,1), (16,1)}|
|(depth, width)|(40,2)|(16,2)|(40,1)|(16,1)|
|:------------:|:----:|:----:|:----:|:----:|
|Teacher       |76.04      |      |      |      |
|Student       |76.04      |70.15      |71.53      |66.30      |
|REFILLED after stage1     |00.00      |00.00      |00.00      |00.00      |
|REFILLED after stage2     |**00.00**  |**00.00**  |**00.00**  |**00.00**  |

**All the results and models will be released soon.**

**Dataset:CUB200**

**Teacher:mobile_net-1.0**

**Student:mobile_net-{1.0,0.75,0.5,0.25}**
|(depth, width)|(40,2)|(16,2)|(40,1)|(16,1)|
|:------------:|:----:|:----:|:----:|:----:|
|Teacher       |76.19      |      |      |      |
|Student       |76.19      |74.49      |72.68      |68.80      |
|REFILLED after stage1     |00.00      |00.00      |00.00      |00.00      |
|REFILLED after stage2     |**00.00**  |**00.00**  |**00.00**  |**00.00**  |

**All the results and models will be released soon.**

## Code and Arguments
This code implements REFILLED under the setting where a source task and a target task is given. **main.py** is the main file and the arguments it take are listed below.

### Task Arguments
- `data_name`: name of dataset
- `teacher_network_name`: architecture of teacher model
- `student_network_name`: architecture of student model
### Experiment Environment Arguments
- `devices`: list of gpu ids
- `flag_gpu`: whether to use gpu or not
- `flag_no_bar`: whether to use a bar
- `n_workers`: number of workers in data loader
- `flag_tuning`: whether to tune the hyperparameters on validation set or train on the whole training set
### Optimizer Arguments
- `lr1`: initial learning rate in stage 1
- `lr2`: initial learning rate in stage 2
- `point`: when to decrease the learning rate
- `gamma`: the extent of learning rate decrease
- `wd`: weight decay
- `mo`: momentum
### Network Arguments
- `depth`: depth of resnet and wide_resnet
- `width`: width of wide_resnet
- `ca`: channel coefficient of mobile_net
- `dropout_rate`: dropout rate of the network
### Training Procedure Arguments
- `n_training_epochs1`: number of training epochs in stage 1
- `n_training_epochs2`: number of training epochs in stage 2
- `batch_size`: batch size in training
- `tau1`: temperature for stochastic triplet embedding in stage 1
- `tau2`: temperature for local distillation in stage 2
- `lambd`: weight of teaching loss in stage 2