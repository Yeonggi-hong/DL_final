# DL Final
## Dependency
    python version 3.8.12
    cuda version 11.4
    pip install requirements.txt
    
## Training
    python training.py --freeze 0 --opt 'ADAM' --lr 0.0001 --sch 'True' --pretrained yes --num_head 4
#### --batch_size   default=700     Batch size
#### --lr           default=0.001   Initial learning rate for adam
#### --workers      default=2       Number of data loading workers
#### --epochs       default=5       Total training epochs
#### --num_head     default=4       Number of attention head
#### --opt                          ADAM, ADAMW, SGD
#### --freeze                       0, 50, 100
#### --sch                          Ture , False
#### --pretrained                   True , False


## Validation 
    python val.py


## Ensemble Validation
    python ensemble_val.py
