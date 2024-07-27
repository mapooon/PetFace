from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.num_epoch = 10
config.warmup_epoch = 0



config.train_csv_list = [
    'data/PetFace/split/cat/train.csv',
    'data/PetFace/split/chimp/train.csv',
    'data/PetFace/split/chinchilla/train.csv',
    'data/PetFace/split/degus/train.csv',
    'data/PetFace/split/dog/train.csv',
    'data/PetFace/split/ferret/train.csv',
    'data/PetFace/split/guineapig/train.csv',
    'data/PetFace/split/hamster/train.csv',
    'data/PetFace/split/hedgehog/train.csv',
    'data/PetFace/split/parakeet/train.csv',
    'data/PetFace/split/javasparrow/train.csv',
    'data/PetFace/split/pig/train.csv',
    'data/PetFace/split/rabbit/train.csv',
]