ts=100
cluster=10
# for dataset in Beauty ML100K ML1M Yelp Electronics Movies CDs Clothing Taobao LastFM
for dataset in Beauty Movies CDs Clothing 
# for dataset in Electronics
do
    # for indexing in random sequential collaborative
    for indexing in random sequential
    do
        python ./src/src_llama/generate_dataset.py --dataset ${dataset} --data_path /shared/share_mala/andrew/OpenP5/data/ --item_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --collaborative_token_size ${ts} --collaborative_cluster ${cluster}

        python ./src/src_llama/generate_dataset_eval.py --dataset ${dataset} --data_path /shared/share_mala/andrew/OpenP5/data/ --item_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode validation --prompt seen:0 --collaborative_token_size ${ts} --collaborative_cluster ${cluster}

        python ./src/src_llama/generate_dataset_eval.py --dataset ${dataset} --data_path /shared/share_mala/andrew/OpenP5/data/ --item_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode test --prompt seen:0 --collaborative_token_size ${ts} --collaborative_cluster ${cluster}

        python ./src/src_llama/generate_dataset_eval.py --dataset ${dataset} --data_path /shared/share_mala/andrew/OpenP5/data/ --item_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode test --prompt unseen:0 --collaborative_token_size ${ts} --collaborative_cluster ${cluster}
    done
done
