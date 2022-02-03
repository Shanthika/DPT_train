# DPT_train
This is the implementation of the paper [Vision Transformers for Dense Prediction](https://openaccess.thecvf.com/content/ICCV2021/papers/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.pdf).

## Installation
Please install conda. Create a new environment and install all the dependencies using the following command
```
conda env create --file environment.yml
```

## Dataset
Use the [link](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) to download the dataset. Or use the command
```
wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
```


## Training the model
The parameters for the model are defined in *config/train.yaml* file. To train the model, run the command
```
python train.py
```

## Testing
The jupyter notebook *eval.ipynb* can be used for testing.




#### Repositories referred:
- [DPT](https://github.com/isl-org/DPT)
- [dynamic-inference](https://github.com/chriswxho/dynamic-inference)
