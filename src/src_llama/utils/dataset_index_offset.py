def get_dataset_offset(dataset_name):
    offsets = {
        "CDs": 100000,
        "Beauty": 200000,
        "Clothing": 300000,
        "Electronics": 400000,
        "Movies": 500000,
        "ML100K": 600000,
        "ML1M": 700000,
        "Yelp": 800000,
        "Taobao": 900000,
        "LastFM": 1000000,
    }
    return offsets.get(dataset_name, 0)
