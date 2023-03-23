# Sequential recommendation with RL

A framework for diverse and accurate recommendation for users in sequential recommendation via reinforcement learning techniques


## Setups
[![Python](https://img.shields.io/badge/python-3.9.12-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.13.0-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)


## Datasets
The dataset name must be specified in the "--dataset" argument
- [Beauty](https://github.com/RUCAIBox/CIKM2020-S3Rec/tree/master/data) 
- [last.fm](https://github.com/RUCAIBox/CIKM2020-S3Rec/tree/master/data)

After downloaded the datasets, you can put them in the folder `data/` like the following.
```
$ tree
.
├── beauty
│   ├── preprocess_beauty.py
│   ├── Beauty.txt
│   └── item2attributes.json
└── lastfm
    ├── preprocess_lastfm.py
    ├── LastFM.txt
    └── item2attributes.json

```

And you can preprocess each datasets by running,
```
python preprocess_{dataset_name}.py
```


## Train and Test
```
python main.py --dataset lastfm --gpu_num 1
```


## Reference
- SA2C, SNQN (WSDM '22) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3488560.3498494) [[code]](https://drive.google.com/file/d/185KB520pBLgwmiuEe7JO78kUwUL_F45t/view?usp=sharing) <br>
- Datasets with attributes [[link]](https://github.com/RUCAIBox/CIKM2020-S3Rec/tree/master/data)


### Project presenstation

[**Presentation PPT (in Korean)**](./asset/SR-RL_final_pt.pdf) <br>
**Presentation Video (in Korean)**
[![Video thumbnail](./asset/SR-RL_final_pt_thumbnail.jpg)](https://www.youtube.com/watch?v=IGXcdc7U_R4)