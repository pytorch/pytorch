#!/bin/sh

cd "$(dirname "$0")"
cd ..

python -u launcher.py \
    --master_addr="localhost" \
    --master_port="29500" \
    --trainer="DdpSparseDenseRpcTrainer" \
    --ntrainer=0 \
    --ncudatrainer=2 \
    --filestore="/tmp/tmpn_k_8so02" \
    --server="AverageBatchParameterServer" \
    --nserver=0 \
    --ncudaserver=1 \
    --rpc_timeout=30 \
    --backend="gloo" \
    --epochs=10 \
    --batch_size=10 \
    --data="DummyData" \
    --model="DummyModelSparse" \
    --data_config_path="configurations/data_configurations.json" \
    --model_config_path="configurations/model_configurations.json"
