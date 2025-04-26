for x in 100; do
    python main.py --save_name "test_lamb${x}" --dim 128 --use_wandb --use_svi --lamb "$x"
done


#1 5 10 20 50 100 200 300 500