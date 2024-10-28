import torch.nn as nn
import ml_collections


def get_train_gen_config():
    """
    Generator parameter configuration.
    """

    config = ml_collections.ConfigDict()
    config.patch_size = 4
    config.use_hscam = True

    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.mamba = ml_collections.ConfigDict()
    config.mamba.input_chans = 3
    config.mamba.output_chans = 1
    config.mamba.encoder_depths = [2, 2, 2, 2]
    config.mamba.decoder_depths = [2, 2, 2, 1]
    config.mamba.embed_dims = [96, 192, 384, 768]

    config.mamba.drop_path_rate = 0.2
    config.mamba.load_ckpt_path = None
    config.mamba.d_state = 16
    config.mamba.drop_rate = 0.
    config.mamba.attn_drop_rate = 0.
    config.mamba.drop_path_rate = 0.1
    config.mamba.norm_layer = nn.LayerNorm
    config.mamba.patch_norm = True
    config.mamba.use_checkpoint = False

    return config


def get_train_disc_config():
    """
    Discriminator parameter configuration.
    """

    config = ml_collections.ConfigDict()
    config.patch_size = 4
    config.use_hscam = True
    # config.transformer.num_layers = 4  # 层数与len(config.mamba.encoder_depths)一致

    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.mamba = ml_collections.ConfigDict()
    config.mamba.input_chans = 3
    config.mamba.output_chans = 1
    config.mamba.encoder_depths = [2, 2, 2, 2]
    config.mamba.decoder_depths = [2, 2, 2, 1]
    config.mamba.embed_dims = [96, 192, 384, 768]

    config.mamba.drop_path_rate = 0.2
    config.mamba.load_ckpt_path = None
    config.mamba.d_state = 16
    config.mamba.drop_rate = 0.
    config.mamba.attn_drop_rate = 0.
    config.mamba.drop_path_rate = 0.1
    config.mamba.norm_layer = nn.LayerNorm
    config.mamba.patch_norm = True
    config.mamba.use_checkpoint = False

    return config


def get_test_config():
    return get_train_gen_config()
