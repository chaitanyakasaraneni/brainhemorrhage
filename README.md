## RSNA Intracranial Hemorrhage Detection

This is a solution to RSNA Intracranial Hemorrhage Detection Competition on [kaggle.com](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)

### Introduction: 
Intracranial Hemorrhage is a brain disease that causes bleeding inside the cranium. This is a serious health issue and the patient having this often requires immediate and intensive treatment. For example, intracranial hemorrhages account for approximately 10% of strokes in the U.S., where stroke is the fifth-leading cause of death. Identifying the location and type of any hemorrhage present is a critical step in treating the patient. Spontaneous intracranial hemorrhage can be related to variety of disease processes including, but not limited to arteriovenous malformations, ruptured aneurysms, anticoagulation, tumors, venous sinus thrombosis, hypertension, cerebral amyloid angiopathy, and hemorrhagic conversion of strokes [1][2][3][4]. Diagnosis requires an urgent procedure. When a patient shows acute neurological symptoms such as severe headache or loss of consciousness, highly trained specialists review medical images of the patient’s cranium to look for the presence, location and type of hemorrhage. The process is complicated and often time consuming. For this reason, deep learning is used to detect the intracranial hemorrhage in a faster way. <br>
There are four major types of intracranial hemorrhage: epidural, subdural, subarachnoid, and intraparenchymal, which refer to the location of bleeding. Unenhanced computed tomography (CT) scans of the brain are commonly used to evaluate for intracranial hemorrhage [5]. Differences in x-ray attenuation and location of intracranial hemorrhage on unenhanced CT scans of the brain make them detectable and allow the different types of intracranial hemorrhage to be differentiated [6]. The main aim of this project is to detect acute intracranial hemorrhage and its subtypes in a single step by applying novel deep learning techniques on the CT scan images provided.

### Related Work:
Intracranial hemorrhage image attenuation significantly overlaps with those of gray matter, meaning that simple thresholding is ineffective [7]. There have been several methods developed for intracranial hemorrhage detection using image processing techniques. These methods follow a traditional approach of detecting head in the image, aligning the head, removing the skull, compensating for cupping CT artifacts, extracting handcrafted features from the imaged brain tissue, and classifying intracranial hemorrhage voxels based on the features. Most have used small datasets of 11–30 cases. In some approaches, manual intervention is sometimes needed, such as for aligning the head [6]. <br>
One of the more sophisticated approaches [6] specifically focuses on detecting small (<1 cm) acute intracranial hemorrhages. In addition to executing the processing steps listed above, this approach also registers the head to an anatomic model and adjusts detections based on anatomic regions. A follow-on study [8] determined that this computer-aided diagnosis system improved detection of these hemorrhages by emergency physicians and radiology residents, although not by radiology specialists.”
#### Data Collection:
There is a dataset available online provided by Research Society of North America (RSNA). This dataset contains over four million train images, a .csv file containing images with the type of acute hemorrhage in a column and probability of the type present in the other column, and over four hundred thousand test images. This total dataset is of about 156 GB. 
Dataset: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data <br>

The dataset consists of DICOM images. All provided images are in DICOM format. DICOM images contain associated metadata. This will include PatientID, StudyInstanceUID, SeriesInstanceUID, and other features. The training data is provided as a set of image Ids and multiple labels, one for each of five sub-types of hemorrhage, plus an additional label for any, which should always be true if any of the sub-type labels is true. There is also a target column, Label, indicating the probability of whether that type of hemorrhage exists in the indicated image. There will be 6 rows per image Id. The label indicated by a particular row will look like [Image Id], [Sub-type Name].
  
### Model Selection:
There are previous papers that employ basic CNNs, LeNet, GoogLeNet, Inception-ResNet to detect brain hemorrhage. But these projects have just considered a few cases and not many. In this project, PyTorch ResNeXt [9] and weakly supervised pretraining are used to detect acute intracranial hemorrhage. ResNext is developed by researchers at UC San Diego in collaboration with Facebook AI Research in 2016. ResNeXt is a simple, highly modularized network architecture for image classification. The ResNeXt network consists of a stack of residual blocks which have the same topology as VGG/ResNet [9]. ResNeXt adopts the same VGG/ResNet strategy of repeating layers while exploiting the split-transform-merge strategy in an easy, extensible way. <br>

“The network is constructed using two rules inspired from VGG/ResNet. They are (i) if producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes), and (ii) each time when the spatial map is down sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2. The second rule ensures that the computational complexity, in terms of FLOPs (floating-point operations, in # of multiply-adds), is roughly the same for all blocks.” [9]<br>

In this project, prior to ResNext, windowing functions were employed to preprocess the images. For this reason, to speed up the process, Nvidia’s Apex library [10] is used for mixed precision training. Some files didn't contain legitimate images, so these images were eliminated as a part of preprocessing. The code for ResNext is publicly available on GitHub. <br>

The input image is of size 224x224, randomly cropped from a resized image using the scale and aspect ratio augmentation. Downsampling of conv3, 4, and 5 is done by stride-2 convolutions in the 3×3 layer of the first block in each stage, as suggested in [11]. SGD with a mini-batch size of 256 on 8 GPUs (32 per GPU) is used. The weight decay is 0.0001 and the momentum is 0.9. Tested starting from learning rate of 0.1 and divided it by 10 for three times using the schedule in [11]. The weight initialization of [12] was adopted. In all ablation comparisons, the error on the single 224×224 center crop from an image whose shorter side is 256 was evaluated.

### Output:
•Id - An image Id. Each Id corresponds to a unique image and will contain an underscore with hemorrhage subtype after the underscore.
•Label - The probability of whether that sub-type of hemorrhage (or any hemorrhage in the case of any) exists in the indicated image.

### References:
[1] N.J. Fischbein, C.A.C. Wijma, "Nontraumatic intracranial hemorrhage", Neuroimaging Clin N Am, vol. 20, no. 4, pp. 469-492, 2010
[2] D.K. Nishijima, S.R. Offerman, D.W. Ballard, S Zehtabchi et al., "Risk of traumatic intracranial hemorrhage in patients with head injury and preinjury Warfarin or Clopidogrel use", Acad Emerg Med, vol. 20, no. 2, pp. 140-145, 2013
[3] S. Sacco, C. Marini, D. Toni, L. Olivieri, A. Carolei, "Incidence and 10-year survival of intracerebral hemorrhage in a population-based registry", Stroke, vol. 40, no. 2, pp. 394-399, 2009
[4] D.B. Zahuranec, N.R. Gonzales, D.L. Brown et al., "Presentation of intracerebral haemorrhage in a community", J Neurol Neurosurg Psychiatry, vol. 77, no. 3, pp. 340-344, 2005
[5] J.C. Hemphill, S.M. Greenberg, C.S. Anderson et al., "Guidelines for the management of spontaneous intracerebral hemorrhage", Stroke, vol. 46, no. 7, pp. 2032-2060, 2015.
[6] T. Chan, "Computer aided detection of small acute intracranial hemorrhage on computer tomography of brain", Comput Med Imaging Graph, vol. 31, no. 4, pp. 285-298, 2007.
[7] W.L Nowinski et al., "Characterization of interventricular and intracerebral intracranial hemorrhages in non-contrast CT", Neuroradiology J., vol. 27, pp. 299-315, 2014.
[8] T. Chan, H. Kuang, "Effect of a computer-aided diagnosis system on clinician’s performance in detection of small acute intracranial hemorrhage on computed tomography", Acad Radiol, vol. 15, pp. 290-299, 2008.
[9] S. Xie, R. Girshick, P. Dollár, Z. Tu and K. He, "Aggregated Residual Transformations for Deep Neural Networks," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017, pp. 5987-5995.
[10] https://github.com/NVIDIA/apex
[11] S. Gross, M. Wilber, Training and investigating Residual Nets, 2016, [online] Available: https://github.com/facebook/fb.resnet.torch.
[12] K. He, X. Zhang, S. Ren, J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ima-genet classification", ICCV, 2015.
- Windowing functions for pre-processed data taken from https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
- https://www.kaggle.com/taindow/pytorch-efficientnet-b0-benchmark
- https://www.kaggle.com/taindow/pytorch-resnext-101-32x8d-benchmark
- https://github.com/facebookresearch/WSL-Images
- https://github.com/NVIDIA/apex
