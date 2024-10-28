if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/Autoformer" ]; then
  mkdir ./logs/LongForecasting/Autoformer
fi

seq_len=96
label_len=48
features=M
gpu=2
model_name=Autoformer

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/traffic \
    --data_path traffic.csv \
    --model_id $norm_type_traffic_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --period_len 24 \
    --pe_layers 0 \
    --pd_ff 512 \
    --pd_model 256 \
    --j 1 \
    --itr 3 >logs/LongForecasting/Autoformer/$model_name'_traf_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/electricity \
    --data_path electricity.csv \
    --model_id $norm_type_electricity_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --period_len 24 \
    --j 1 \
    --pe_layers 0 \
    --itr 3 >logs/LongForecasting/Autoformer/$model_name'_elec_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/weather \
    --data_path weather.csv \
    --model_id $norm_type_weather_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --period_len 12 \
    --j 1 \
    --itr 3 >logs/LongForecasting/Autoformer/$model_name'_wea_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh1.csv \
    --model_id $norm_type_ETTh1_96_$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 24 \
    --j 1 \
    --twice_epoch 3 \
    --itr 3 >logs/LongForecasting/Autoformer/$model_name'_ETTh1_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh2.csv \
    --model_id $norm_type_ETTh2_96_$pred_len \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --gpu 0 \
    --period_len 24 \
    --kernel_len 32 \
    --twice_epoch 0 \
    --pd_ff 128 \
    --pd_model 128 \
    --pe_layers 0 \
    --itr 3 >logs/LongForecasting/Autoformer/$model_name'_ETTh2_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTm1.csv \
    --model_id $norm_type_ETTm1_96_$pred_len \
    --model $model_name \
    --data ETTm1 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --hkernel_len 7 \
    --pd_ff 128 \
    --j 1 \
    --itr 3 >logs/LongForecasting/Autoformer/$model_name'_ETTm1_'$pred_len.log
  done

for pred_len in 336 720 96 192; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTm2.csv \
    --model_id $norm_type_ETTm2_96_$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --twice_epoch 0 \
    --pe_layers 1 \
    --pd_ff 512 \
    --dr 0.2 \
    --itr 3 >logs/LongForecasting/Autoformer/$model_name'_ETTm2_'$pred_len.log
  done
