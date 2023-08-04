python -u main.py \
    --is_training 0 \
    --root_path ./dataset/competition/train-5min/ \
    --data_path flow.csv \
    --d_model 16 \
    --d_ff 16 \
    --gpu 0 \
    --num_kernels 3 \
    --top_k 3 \
    --lradj type3 \
    --patience 20 \
    --freq h \
    --learning_rate 0.001\
    --batch_size 32\
    --e_layers 2\
    --dropout 0.2

python -u main.py \
    --is_training 0 \
    --root_path ./dataset/competition/train-5min/ \
    --data_path flow-5min.csv \
    --d_model 32 \
    --d_ff 32 \
    --gpu 0 \
    --num_kernels 6 \
    --top_k 5 \
    --lradj type3 \
    --patience 20 \
    --freq h \
    --learning_rate 0.001\
    --batch_size 16\
    --e_layers 2

python -u main.py \
    --is_training 0 \
    --root_path ./dataset/competition/train-5min/ \
    --data_path flow.csv \
    --d_model 64 \
    --d_ff 64 \
    --gpu 0 \
    --num_kernels 6 \
    --top_k 5 \
    --lradj type3 \
    --patience 20 \
    --freq h \
    --learning_rate 0.001\
    --batch_size 16\
    --e_layers 2

python -u main.py \
    --is_training 0 \
    --root_path ./dataset/competition/train-5min/ \
    --data_path flow.csv \
    --d_model 128 \
    --d_ff 128 \
    --gpu 0 \
    --num_kernels 6 \
    --top_k 5 \
    --lradj type3 \
    --patience 10 \
    --freq h \
    --learning_rate 0.0001\
    --batch_size 16\
    --e_layers 2


python -u main.py \
    --is_training 0 \
    --root_path ./dataset/competition/train-5min/ \
    --data_path flow.csv \
    --d_model 256 \
    --d_ff 128 \
    --gpu 0 \
    --num_kernels 6 \
    --top_k 5 \
    --lradj type3 \
    --patience 10 \
    --freq h \
    --learning_rate 0.0001\
    --batch_size 16\
    --e_layers 2