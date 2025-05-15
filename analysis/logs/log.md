# Weekly/Bi-Weekly Log

## Prompts
Following the [Rose/Bud/Thorn](https://www.panoramaed.com/blog/rose-bud-thorn-activity-and-worksheet#:~:text=%22Rose%2C%20Bud%2C%20Thorn%22%20is%20a%20mindful%20design%2D,day%2C%20week%2C%20or%20month.) model:

## Date: Week 9 (Mar 14, 2025) 



### Number of hours: 
~10 hours (problem statement selection, dataset prep, initial model setup)

### Rose:
* Successfully set up the MONAI framework and started model development for abnormality detection in brain CT scans using the MedTrinity-25M demo dataset hoping to get access to the full dataset upon request.

* Completed data preprocessing, augmentation, and created efficient PyTorch dataloaders for training.
### Bud: 
* Looking forward to scaling up to the full 25M dataset and improving model accuracy.

* Planning to integrate an LLM for medical report generation.
### Thorn: 
* Full dataset access is delayed due to an expired download link and the accessible dataset is only for brain images. 

* Initial results show 70–80% accuracy, but further improvement is needed.

* Communication and clear weekly planning need to be improved to avoid slowdowns.



## Date: Week 10 (Mar 21, 2025) 



### Number of hours: 
~12 hours (model training, team meetings)

### Rose:
* Changed model approach from UNet to SwinUNETR for better performance on medical images.
* Began implementing anomaly detection with MSE loss and image reconstruction.
### Bud: 
* Exploring new datasets to overcome current limitations; requests sent to Synapse for access to more comprehensive data with validation sets.

* Considering other models and architectures for improved segmentation and classification.
### Thorn: 
* Dataset issues: lack of validation set and only anomaly instances, which could bias the model.

* Computational expense and time required for image data processing remain significant challenges.


## Date: Week 11 (Mar 28, 2025) 



### Number of hours: 
~10 hours (dataset research, model testing)

### Rose:
* Implemented SwinUNETR with the BRATS dataset using MONAI, leveraging prebuilt models for brain tumor segmentation.
* Team members assigned to research and vet additional datasets to address current data gaps.

### Bud: 
* Investigating PubmedCLIP and MedCLIP for potential use in the NLP/report generation component.
* Aiming to establish a robust pipeline for both segmentation and classification tasks.
### Thorn: 
* Dataset diversity: different scan types and lack of normal cases complicate model generalization.

* Still waiting for access to larger, more representative datasets.



## Date: Week 12 (Apr 4, 2025) 



### Number of hours: 
~8 hours (model development, literature review)

### Rose:
* Developed a generic image classification model for medical imaging tasks.
* Successfully implemented a sample brain tumor segmentation, achieving deeper understanding of scan processing.

### Bud: 
* Plan to refine segmentation accuracy, possibly by adjusting loss functions (e.g., Dice loss) and experimenting with more advanced architectures (UNet++, etc.).
* Looking into state-of-the-art multimodal models for future report generation.
### Thorn: 
* Model performance is still limited by data quality and size.

* Need to improve class balancing and address class imbalance in rare disease detection.


## Date: Week 13 (Apr 11, 2025) 



### Number of hours: 
~9 hours (segmentation model, code review)

### Rose:
* Implemented and tested a UNet++ segmentation model, achieving ~80% accuracy on tumor masks.
* Improved documentation and code organization in the GitHub repository.

### Bud: 
* This week, we gained access to the MIMIC-CXR dataset from PhysioNet, which meets our requirements for scale, label quality, and real-world relevance.
* Decided to pivot the project focus to chest disease classification and automated report generation using MIMIC-CXR.
* Our new goal: enable users to upload chest X-rays and receive simple, accessible reports in everyday language.
### Thorn: 
* Limited by computational resources and GPU memory for larger models and batch sizes.

* Some segmentation outputs still lack clinical coherence.

## Date: Week 14 (Apr 18, 2025) 



### Number of hours: 
~10 hours (chest disease classification with DenseNet-121, established baseline for automated report generation)

### Rose:
* Launched multi-label chest disease classification using DenseNet-121 as the backbone on the MIMIC-CXR dataset, leveraging robust transfer learning and advanced data augmentation.

* Achieved strong accuracy and F1 scores for common diseases (e.g., Atelectasis, Support Devices), with clear per-disease metrics tracked for accuracy, F1, and AUC.

* Established a baseline for automated chest X-ray report generation using a BiomedCLIP (ViT) image encoder and BioGPT decoder, with initial ROUGE-L = 0.32 on test samples.

### Bud: 
* Plan to implement class balancing strategies (weighted loss, oversampling) to address the impact of class imbalance on rare disease detection.

* Next, develop an advanced automated report generation module using transformer-based or multimodal architectures (e.g., ChestBioX-Gen with BioGPT and co-attention, or Flamingo-CXR), aiming for clinically meaningful, patient-friendly reports directly from chest X-rays.

* Explore entity-aware fine-tuning and improved decoding strategies (like beam search) to enhance report coherence and accessibility for non-experts.


### Thorn: 
* Class imbalance in the dataset leads to misleadingly high accuracy for rare diseases (e.g., Pneumothorax: high accuracy, low F1).

* Current report outputs are fragmented, repetitive, and lack clinical detail or accessible language, highlighting the need for better image/text embedding alignment and more robust, patient-oriented report generation.




## Date: Week 15 (Apr 25, 2025) 



### Number of hours: 
~11 hours (model refinement, performance evaluation, deployment planning)

### Rose:
* Achieved reproducible, scalable deep learning experimentation pipeline for chest disease classification and report generation, with robust per-disease metrics (accuracy, F1, AUC) on the MIMIC-CXR dataset.

* Integrated best practices for data pipeline management, including dynamic loading, advanced augmentation, and mixed-precision training for efficiency and generalization.

* Switched to multi-class classification with DenseNet-121 instead of multi-label classification, as many images have multiple labels, making training more structured and manageable.

### Bud: 
* Planning to implement a working vision-text transformer model like BiomedCLIP, which has demonstrated state-of-the-art performance with biomedical images and can effectively handle the medical terminology in chest X-ray reports.

* Continue developing and refining the automated report generation module, focusing on transformer-based models (e.g., ChestBioX-Gen, Flamingo-CXR) to produce clinically relevant and patient-accessible radiology reports.

* Explore integration of entity-aware fine-tuning and improved decoding strategies (like beam search) to enhance report coherence and accessibility for non-experts.

### Thorn: 
* Training separate models for each disease is resource-intensive and ignores disease correlations; a unified multi-label model is needed for better efficiency and clinical realism.

* Accuracy scores are compromised due to significant label imbalance in our current dataset (only 50GB of the full 500GB MIMIC-CXR dataset), resulting in overrepresentation of certain conditions and underrepresentation of others.

* Complex dataset structure complicates pipeline development: multiple CSV files (metadata, CheXpert labels) need to be joined with JPG images located in separate folders and corresponding text reports in different files.


## Additional thought
Write anything that you think would be important for YOU later on.

---

