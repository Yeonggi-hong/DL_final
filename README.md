# DL Final
## Dependency
    python version 3.8.12
    pip install requirements.txt

## Training
    python training.py --freeze 0 --opt 'ADAM' --lr 0.0001 --sch 'True' --pretrained yes --num_head 4
#### '--batch_size', type=int, default=700, help='Batch size.'
#### '--lr', type=float, default=0.001, help='Initial learning rate for adam.'
#### '--workers', default=2, type=int, help='Number of data loading workers.'
#### '--epochs', type=int, default=5, help='Total training epochs.'
#### '--num_head', type=int, default=4, help='Number of attention head.'
#### '--opt', type=str
#### '--freeze', type=str
#### '--sch', type=str
#### '--pretrained', type=str


## Val 
    python val.py


## Ensemble Val
    python ensemble_val.py