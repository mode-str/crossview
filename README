# MCCG

This repository contains the dataset link and the code for our paper *MCCG: A ConvNeXt-based Multiple-Classifier
Method for Cross-view Geo-localization*. Thank you for your kindly attention.

## requirement
1. Download the [University-1652](https://github.com/layumi/University1652-Baseline) dataset
2. Download the [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark) dataset
3. Configuring the environment
   * First you need to configure the torch and torchision from the [pytorch](https://pytorch.org/) website
   * ```shell
     pip install -r requirement.txt
     ```

## About dataset
The organization of the dataset.

More detailed about Univetsity-1652 dataset structure:
```
├── University-1652/
│   ├── train/
│       ├── drone/                   /* drone-view training images 
│           ├── 0001
│           ├── 0002
│           ...
│       ├── street/                  /* street-view training images 
│       ├── satellite/               /* satellite-view training images       
│       ├── google/                  /* noisy street-view training images (collected from Google Image)
│   ├── test/
│       ├── query_drone/  
│       ├── gallery_drone/  
│       ├── query_street/  
│       ├── gallery_street/ 
│       ├── query_satellite/  
│       ├── gallery_satellite/ 
│       ├── 4K_drone/
```
More detailed about SUES-200 dataset structure:
```
├── SUES-200/
│   ├── train/
│       ├── 150/
│           ├── drone/                   /* drone-view training images 
│               ├── 0001
│               ├── 0002
│               ...
│           ├── satellite/               /* satellite-view training images       
│       ├── 200/                  
│       ├── 250/  
│       ├── 300/  
│   ├── test/
│       ├── 150/  
│           ├── query_drone/  
│           ├── gallery_drone/  
│           ├── query_satellite/  
│           ├── gallery_satellite/ 
│       ├── 200/  
│       ├── 250/  
│       ├── 300/  
```


## Train and Test
We provide scripts to complete MCCG training and testing
* Change the **data_dir** and **test_dir** paths in **run.sh** and then run:
```shell
bash run.sh
```
