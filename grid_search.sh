for x in 50 75 100 125 150 175; do
    echo "Running with radius = $x"
    python main.py --save_name "grid_radius" --dim 64 --use_wandb --use_svi --radius "$x"
done



#1 5 10 20 50 100 200 300 500
#python main.py --use_wandb --use_svi --dim 128 --save_name ln