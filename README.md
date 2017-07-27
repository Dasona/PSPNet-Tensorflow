# PSPNet-Tensorflow
Tensorflow Semantic Segmentation Workbench for RGBD/RGB Images. Algorithm implemented : PSPNet (https://github.com/hszhao/PSPNet)

Important information regarding the scripts and their parameters is available from this repo's [Wiki](https://github.com/amanjhunjhunwala/PSPNet-Tensorflow/wiki/Description-of-scripts-and-parameters)

`Requirements:

tensorflow-gpu-0.12rc1 (The repo is not comptible with Tensorflow 1.+) (will be in a few days, hopefully!)

+ Properly set Anaconda python installation `

### Workflow of this repository using ADE20K dataset (other datasets are similar) :

1. First download the dataset and extract the contents into the training set directory and validation directory as mentioned in the Wiki !

2. The pixel values of annotation images are changed from 1-150 to 0-149 to eliminate void class. Void class leads to lower accuracy 

3. Run data conversion script follwoing the 1st script parameter list in the Wiki

4. Check to see if records are generated in the directory

5.0 Download the converted weights for intialization from [here](http://bit.do/PSP-TF-Initialization) 

Make sure to add the path to this directory as the checkpoint path ! 

5.1 Run the training.sh script after setting the flags to relevant values

6. Verify in Tensorboard if the Images and Labels actually show, in the corresponding tab

7. Once training is complete, use the evaluation.sh script for performance evaluation

8. Freeze the model using the freeze_model.sh script 

9. Run demo.sh to start an online demo. (Note URL at which the website would be available would be /ui/segmentation)

10. Check and see the results visually !


NOTES :

1. The auxilary branch for training has not been added yet. It will, hopefully be, in a few days !
2. Pull requests most welcome 
