"""
authors: Gabriel Nobis & Maximilian Springenberg
copyright: Fraunhofer HHI
"""

import model_zoo


def get_model(
    model_name,
    image_size,
    channels,
    model_channels,
    num_res_blocks,
    attn_resolutions,
    num_classes,
    channel_mult,
    dropout,
    *args,
    **kwargs
):

    if model_name.lower() == "unet":
        model = model_zoo.UNetModel(
            image_size=image_size,
            in_channels=channels,
            model_channels=model_channels,
            out_channels=channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attn_resolutions,
            num_classes=num_classes,
            channel_mult=channel_mult,
            dropout=dropout,
        )
    else:
        NotImplementedError

    return model