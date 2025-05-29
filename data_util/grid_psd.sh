for x in 50 75 100 125 150 175; do
    echo "Preprocess with psd = $x"
    python data_util/preprocess.py --radius "$x"
done