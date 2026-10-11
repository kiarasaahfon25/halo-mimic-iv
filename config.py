'''
    code by Brandon Theodorou
    Original GPT-2 Paper and repository here: https://github.com/openai/gpt-2
    Original GPT-2 Pytorch Model: https://github.com/huggingface/pytorch-pretrained-BERT
    GPT-2 Pytorch Model Derived From: https://github.com/graykode/gpt-2-Pytorch

 HALO Configuration
   - total_vocab_size: total number of available tokens 
   - code_vocab_size: number of available medical codes 
   - label_vocab_size: number of available label codes
   - special_vocab_size: number of special tokens (START, END, PAD)
   - n_positions: number of positions available in the input sequence
   - n_ctx: amount of sequence information HALO can use as context
   - n_embd: size of embedding for each token
   - n_layer: number of Transformer blocks
   - n_head: number of attention heads in each Transformer block
   - layer_norm_epsilon: used for stability during normalization
   - initializer_range: controls the range used to initialize model weights
   - batch_size: number of training examples processed at once
   - sample_batch_size: number of examples processed at once during sampling
   - epoch: number of times the model goes through the training dataset
   - pos_loss_weight: controls whether positive examples receive additional weight in the loss
   - lr: learning rate
'''

class HALOConfig(object):
    def __init__(
            self,
            total_vocab_size=6869, 
            code_vocab_size=6841,   #medical codes to choose from original was 6841
            label_vocab_size=25,    #label codes to choose from
            special_vocab_size=3,   #start code, end code, pad code
            n_positions=56,  #56 positions available in the sequence
            n_ctx=48,  #how much sequence information can be used as context
            n_embd=768,  #embedding size??
            n_layer=12,  #transformer blocks
            n_head=12,   #12 attention heads
            layer_norm_epsilon=1e-5,  #normalization setting (not to be worried about much)
            initializer_range=0.02, #weight initialization
            batch_size=48,
            sample_batch_size=256,
            epoch= 50, #original was 50, reducing for tests
            pos_loss_weight=None,  #controls if positive examples are given more weight
            lr=1e-4,
    ):
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.label_vocab_size = label_vocab_size
        self.special_vocab_size = special_vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.sample_batch_size = sample_batch_size
        self.epoch = epoch
        self.pos_loss_weight = pos_loss_weight
        self.lr = lr