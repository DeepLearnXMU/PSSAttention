# PSSAttention

**This project is used as students' homework to test their reimplement abilities.**

**Thus, the relevant codes are shielded for a short time (until 2022.02.01).**

**Other researchers, who want to access this project, please feel free to contact us via e-mail: jssu@xmu.edu.cn and jialong2019@iscas.ac.cn.**


Codes for # ACL2019 paper "Progressive Self-Supervised Attention Learning for Aspect-Level Sentiment Analysis", which contains TNet-Att(+AS) and MN(+AS)

and Codes for # AI 2021 paper "Enhanced Aspect-Based Sentiment Analysis Models with Progressive Self-supervised Attention Learning", which contains Bert.

Copy from: https://github.com/TangJiaLong/PSSAttention

# Bert(+AS)
The usages are the same as TNet-Att(+AS). 

## Citation
If the code is used in your research, please cite our paper as follows:
```
@article{Tang:AI2021,
  author={Jinsong Su, Jialong Tang, Hui Jiang, Ziyao Lu, Yubin Ge, Linfeng Song, Deyi Xiong, Le Sun, Jiebo Luo},
  title={Enhanced Aspect-Based Sentiment Analysis Models with Progressive Self-supervised Attention Learning},
  year={2021},
  journal={Artificial Intelligence}
}
```

# TNet-Att(+AS)

### Update
Load pretrained model or Save trained model.

## Requirements
* Python 3.6
* Theano 0.9.0
* numpy 1.13.1
* pygpu 0.6.9
* GloVe.840B.300d

## Running
```
THEANO_FLAGS="device=gpu0" python main_total.py -ds_name [YOUR_DATASET_NAME] -log_name [YOUR_LOG_NAME]
```

## Citation
If the code is used in your research, please cite our paper as follows:
```
@inproceedings{Tang:ACL2019,
  author={Jialong Tang, Ziyao Lu, Jinsong Su, Yubin Ge, Linfeng Song, Le Sun, Jiebo Luo},
  title={Progressive Self-Supervised Attention Learning for Aspect-Level Sentiment Analysis},
  year={2019},
  booktitle={ACL}
}
```

## Note
Most of this code and data are borrowed from:
```
@inproceedings{Li:ACL2018,
  author={Li, Xin and Bing, Lidong and Lam, Wai and Shi, Bei},
  title={Transformation Networks for Target-Oriented Sentiment Classification},
  year={2018},
  booktitle={ACL}
}
```

# MN(+AS)

## Requirements
* Python 2.7.0 or higher
* TensorFlow 1.6.0 or higher
* GloVe.840B.300d

## Set envirnment variables to enable the GPU support
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/PATH/TO/MN(+AS):$PYTHONPATH
```

## Data
'''
Use MN(+AS)/scripts/build_vocab.py to generate Vocab
```
python MN(+AS)/scripts/build_vocab.py /PATH/TO/DATA  /PATH/TO/Vocab 
```

## Running

1. Use MN(+AS)/bin/train.py to train baseline models for futher erasing and mining.
```
python MN(+AS)/bin/train.py \
       --input /PATH/TO/TRAIN_SET_i/context /PATH/TO/TRAIN_SET_i/aspect /PATH/TO/TRAIN_SET_i/polarity  \
       --validation /PATH/TO/VAL_SET|TEST_SET/context /PATH/TO/VAL_SET|TEST_SET/aspect /PATH/TO/VAL_SET|TEST_SET/polarity  \
       --vocabulary /PATH/TO/Vocab \
       --model BL_MN \
       --parameters=device_list=[0],train_steps=20000,hops=1 \
       --pretrained_embedding /PATH/TO/PRETRAINED_EMBEDDING \
       --output /PATH/TO/TRAINED_MODEL_i
```
2. Use MN(+AS)/bin/predicter.py to get the attention weights of training instances.
```
python MN(+AS)/bin/predicter.py \
    --input /PATH/TO/TRAIN_SET_i/context /PATH/TO/TRAIN_SET_i/aspect /PATH/TO/TRAIN_SET_i/polarity \
    --vocabulary /PATH/TO/Vocab \
    --models BL_MN \
    --checkpoints /PATH/TO/TRAINED_MODEL_i \
    --parameters=predict_batch_size=32,device_list=[0],hops=1 \
    --output /PATH/TO/LOG
```
3. Use MN(+AS)/scripts/erasing_data.py to erase data to get TRAIN_SET_(i+1).
```
python MN(+AS)/scripts/erasing_data.py
```
4. Back to step 1, until i==5.

5. Use MN(+AS)/bin/final_train.py to train final model.
```
python SEMEVAL/thumt/bin/final_train.py \
    --input /PATH/TO/TRAIN_SET/context /PATH/TO/TRAIN_SET/aspect /PATH/TO/TRAIN_SET/polarity /PATH/TO/TRAIN_SET/AS
    _value /PATH/TO/TRAIN_SET/polarity/AS_mask \
    --validation /PATH/TO/VAL_SET|TEST_SET/context /PATH/TO/VAL_SET|TEST_SET/aspect /PATH/TO/VAL_SET|TEST_SET/polarity \
    --vocabulary /PATH/TO/Vocab \
    --model FINAL_BL_MN \
    --parameters=device_list=[0],train_steps=10000,hops=1 \
    --pretrained_embedding /PATH/TO/PRETRAINED_EMBEDDING \
    --output /PATH/TO/TRAINED_MODEL_i
```

## Citation
If the code is used in your research, please cite our paper as follows:
```
@inproceedings{Tang:ACL2019,
  author={Jialong Tang, Ziyao Lu, Jinsong Su, Yubin Ge, Linfeng Song, Le Sun, Jiebo Luo},
  title={Progressive Self-Supervised Attention Learning for Aspect-Level Sentiment Analysis},
  year={2019},
  booktitle={ACL}
}
```

## Note
Most of this code and data are borrowed from:
```
https://github.com/THUNLP-MT/THUMT
```
