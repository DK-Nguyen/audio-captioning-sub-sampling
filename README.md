# Temporal Sub-sampling of Audio Feature Sequences for Audio Captioning 

Set up the project by following the instructions from https://github.com/audio-captioning/dcase-2020-baseline.

To conduct an experiment using the sub-sampling for Audio Captioning, 
run 

```
python main.py -c main_settings -j 0 -d settings/subsampling4/no_attn_lr_1e-4_loss_thr_1e-3 -v 
```

### Settings for the baseline model

The file `settings/subsampling4/no_attn_lr_1e-4_loss_thr_1e-3/model.yaml` holds the settings for the baseline DNN:
    
    use_pre_trained_model: No
    encoder:
        input_dim_encoder: 64
        hidden_dim_encoder: 256
        output_dim_encoder: 256
        dropout_p_encoder: .25
        sub_sampling_factor_encoder: 4
    decoder:
        output_dim_h_decoder: 256
        nb_classes:  # Empty, to be filled automatically.
        dropout_p_decoder: .25
        max_out_t_steps: 22
        mode: 0 # mode 0 for no attention, mode 1 for attention
        num_attn_layers: 0 # number of layers if using attention
        first_attn_layer_output_dim: 0
        
The `use_pre_trained_model` flag indicates if a pre-trained model will be used. If
this flag is set to `Yes`, then the name of the file with the weights of the pre-trained
model has to be specified in the `settings/dirs_and_files.yaml` file. 
 
The `encoder` block has the settings for the encoder of the sub-sampling DNN:

  * the input dimensionality to the first layer of the encoder - `input_dim_encoder`
  * the hidden output dimensionality of the first and second layers of the encoder -
  `hidden_dim_encoder`
  * the output dimensionality of the third layer of the encoder - `output_dim_encoder`
  * the dropout probability for the encoder - `dropout_p_encoder`
  * the sub-sampling factor for the encoder -  `sub_sampling_factor_encoder`

Similarly, the `decoder` block holds the settings for the decoder of the baseline DNN: 

  * the output dimensionality of the RNN of the decoder - `output_dim_h_decoder`
  * the amount of classes for the classifier (it is filled automatically by the
  baseline system) - `nb_classes`
  * the dropout probability for the decoder - `dropout_p_decoder`
  * the maximum output time-steps for the decoder - `max_out_t_steps`
  * mode 0 for no attention in the decoder, mode 1 for using attention - `mode`
  * number of linear layers if using attention - `num_attn_layers`
  * the output dimensionality of the first layer in the attention mechanism
  `first_attn_layer_output_dim`