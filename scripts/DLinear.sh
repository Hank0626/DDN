if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/DLinear" ]; then
  mkdir ./logs/LongForecasting/DLinear
fi

gpu=2
features=M
model_name=DLinear

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/electricity \
    --data_path electricity.csv \
    --model_id electricity_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --station_lr 0.0001 \
    --period_len 24 \
    --j 1 \
    --pd_ff 512 \
    --pd_model 256 \
    --itr 3 >logs/LongForecasting/DLinear/$model_name'_elc_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/traffic \
    --data_path traffic.csv \
    --model_id traffic_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 3 \
    --period_len 24 \
    --learning_rate 0.0005 \
    --j 1 \
    --kernel_len 12 >logs/LongForecasting/DLinear/$model_name'_tra_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/weather \
    --data_path weather.csv \
    --model_id weather_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 3 \
    --period_len 12 \
    --j 1 \
    --twice_epoch 2 \
    --pd_ff 128 \
    --pd_model 128 >logs/LongForecasting/DLinear/$model_name'_wea_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1_336_$pred_len$model_name \
    --model $model_name \
    --data ETTh1 \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 24 \
    --j 1 \
    --pe_layers 0 \
    --pd_model 128 \
    --pd_ff 128 >logs/LongForecasting/DLinear/$model_name'_eh1_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh2.csv \
    --model_id ETTh2_336_$pred_len$model_name \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 24 \
    --pe_layers 0 \
    --pd_model 128 \
    --pd_ff 128 \
    --itr 3 >logs/LongForecasting/DLinear/$model_name'_eh2_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTm1.csv \
    --model_id ETTm1_336_$pred_len$model_name \
    --model $model_name \
    --data ETTm1 \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --j 1 \
    --kernel_len 12 \
    --pd_ff 128 \
    --pd_model 128 \
    --itr 3 >logs/LongForecasting/DLinear/$model_name'_em1_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTm2.csv \
    --model_id ETTm2_336_$pred_len$model_name \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --pd_ff 32 \
    --pd_model 32 \
    --itr 3 >logs/LongForecasting/DLinear/$model_name'_em2_'$pred_len.log
  done