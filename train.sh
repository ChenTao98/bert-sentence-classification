nohup python train.py --save_model_path train_bert --model bert --device cuda:1 --train --num_epochs 3 --batch_size 20 --lr 5e-5 > train_bert2.out &
nohup python train.py --save_model_path train_roberta --model roberta --device cuda:1 --train --num_epochs 3 > train_roberta.out &
nohup python train.py --save_model_path train_albert --model albert --device cuda:0 --train --num_epochs 3 > train_albert.out &
nohup python train.py --save_model_path train_gpt2 --model gpt2 --device cuda:0 --train --num_epochs 3 > train_gpt2.out &