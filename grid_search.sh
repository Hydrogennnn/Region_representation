for x in 0.0 0.1 0.2 0.3 0.4; do
    echo "Running with svi_drop = $x"
    python main.py --save_name "grid_drop" --dim 128 --use_wandb --use_svi --svi_drop "$x"
done



#1 5 10 20 50 100 200 300 500
#python main.py --use_wandb --use_svi --dim 128 --save_name ln