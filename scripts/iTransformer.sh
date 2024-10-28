if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/iTransformer" ]; then
  mkdir ./logs/LongForecasting/iTransformer
fi

gpu=1
features=M
model_name=iTransformer

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
    --seq_len 720 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --d_ff 512 \
    --d_model 512 \
    --e_layers 3 \
    --des 'Exp' \
    --learning_rate 0.0005 \
    --batch_size 16 \
    --period_len 24 \
    --kernel_len 32 \
    --twice_epoch 2 \
    --j 1 \
    --pd_ff 512 \
    --pe_layers 1 \
    --itr 3 >logs/LongForecasting/iTransformer/$model_name'_elc_'$pred_len.log
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
    --seq_len 720 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --d_ff 512 \
    --d_model 512 \
    --e_layers 4 \
    --des 'Exp' \
    --itr 3 \
    --period_len 24 \
    --kernel_len 24 \
    --hkernel_len 12 \
    --learning_rate 0.001 \
    --twice_epoch 2 \
    --batch_size 16 \
    --j 1 \
    --pd_model 512 \
    --pd_ff 512  >logs/LongForecasting/iTransformer/$model_name'_tra_'$pred_len.log
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
    --seq_len 720 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --d_ff 512 \
    --d_model 512 \
    --e_layers 3 \
    --des 'Exp' \
    --itr 3 \
    --period_len 12 \
    --j 1 \
    --twice_epoch 2 \
    --pd_ff 128 \
    --pd_model 128 >logs/LongForecasting/iTransformer/$model_name'_wea_'$pred_len.log
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
    --seq_len 720 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_ff 128 \
    --d_model 128 \
    --e_layers 2 \
    --des 'Exp' \
    --period_len 24 \
    --j 1 \
    --pe_layers 0 \
    --pd_model 128 \
    --pd_ff 128 \
    --itr 3 >logs/LongForecasting/iTransformer/$model_name'_eh1_'$pred_len.log
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
    --seq_len 720 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_ff 128 \
    --d_model 128 \
    --e_layers 2 \
    --des 'Exp' \
    --period_len 24 \
    --pe_layers 0 \
    --pd_model 128 \
    --pd_ff 128 \
    --itr 3 >logs/LongForecasting/iTransformer/$model_name'_eh2_'$pred_len.log
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
    --seq_len 720 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_ff 128 \
    --d_model 128 \
    --e_layers 2 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --j 1 \
    --pe_layers 0 \
    --twice_epoch 3 \
    --itr 3 >logs/LongForecasting/iTransformer/$model_name'_em1_'$pred_len.log
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
    --seq_len 720 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_ff 128 \
    --d_model 128 \
    --e_layers 2 \
    --des 'Exp' \
    --period_len 12 \
    --pe_layers 0 \
    --itr 3 >logs/LongForecasting/iTransformer/$model_name'_em2_'$pred_len.log
  done
