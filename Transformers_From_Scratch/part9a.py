
from part8g import *

model = Transformer(
    encoder=encoder_train, decoder=decoder_train,
    src_embed=src_embed_train, tgt_embed=tgt_embed_train,
    src_pos=src_pos_train, tgt_pos=tgt_pos_train,
    projection_layer=proj_train
).to(device) # Move model to device

model.eval()
