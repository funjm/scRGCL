# for ds in Pollen Chung Baron1 Baron2 Baron3 Baron4 Muraro Klein Romanov Quake_10x_Bladder Quake_10x_Limb_Muscle Quake_Smart-seq2_Diaphragm Quake_Smart-seq2_Limb_Muscle Quake_Smart-seq2_Trachea Quake_10x_Spleen; do
for ds in Quake_10x_Spleen; do
    echo "=== Running dataset: $ds ==="
    python scripts/hyperparameter_sensitivity.py --sensitivity_dataset "$ds" --param temperature --dropout_equal --seed 100
done