'''
    code by Brandon Theodorou
    Original GPT-2 Paper and repository here: https://github.com/openai/gpt-2
    Original GPT-2 Pytorch Model: https://github.com/huggingface/pytorch-pretrained-BERT
    GPT-2 Pytorch Model Derived From: https://github.com/graykode/gpt-2-Pytorch
'''
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#transform output of neural network (helps the neural network decide how strongly its learned signals should continue through the network.)  - nonlinear
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

'''neural network component - LayerNorm '''
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) #how much an input matters, initially all are 1 (like m in y = mx + b)
        self.bias = nn.Parameter(torch.zeros(hidden_size))  #shifts weigth up and down, initially 0 (like b in y = mx + b)
        self.variance_epsilon = eps #prevents division by zero

    def forward(self, x):
        
        u = x.mean(-1, keepdim=True) #mean
        s = (x - u).pow(2).mean(-1, keepdim=True) #variance
        x = (x - u) / torch.sqrt(s + self.variance_epsilon) #normalization
        return self.weight * x + self.bias #apply learnable weight and bias

#learnable linear transformation 
class Conv1D(nn.Module):
    def __init__(self, nf, nx):  #nf - number of outputs, nx - number of inputs
        super(Conv1D, self).__init__() #Set up this class as a pytorch neural-network module
        self.nf = nf
        w = torch.empty(nx, nf) #empty matrix with nx rows × nf columns
        nn.init.normal_(w, std=0.02) #initialize with random weights with small standard deviation of 0.02.
        self.weight = nn.Parameter(w) #weights are parameters that the model should learn
        self.bias = nn.Parameter(torch.zeros(nf)) #create bias for each output if nf = 3, bias = [0, 0, 0], this bias is also learnable

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,) #the last item in the tensor should be replaced with the number of outputs
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)  #x.view(-1, x.size(-1)) flattens to 2 dimensions - - - - output = input × bias + weight  
        x = x.view(*size_out) #go back to the original structure 
        return x
#attention layer
class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert n_state % config.n_head == 0 #make sure dta can be split evenly among the attention heads
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)) #handles masking (can see itself and past but not future)
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx) #x3 since attention creates Query (What information am i lookin at), key (is this key important? ), value (What is it's value? )
        self.c_proj = Conv1D(n_state, nx) #Output projection: mixes the information from the attention heads together and produces the final attention output.


    def _attn(self, q, k, v):
         # w = attention score matrix (attention weights tell us how much to pay attention to each position)
        w = torch.matmul(q, k) #how much do the queries and the keys match
        if self.scale:
            w = w / math.sqrt(v.size(-1)) #keeps the attention scores at a reasonable size.
        nd, ns = w.size(-2), w.size(-1) #nd = number of rows in the attention score, ns = number of columns in the attention score
        b = self.bias[:, :, ns-nd:ns, :ns] #get mask 
        w = w * b - 1e10 * (1 - b) #apply mask 
        w = nn.Softmax(dim=-1)(w) #Softmax converts the scores into probabilities/attention weights.
        return torch.matmul(w, v) #final attention output

    #combine 12 attentions heads
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)
    
    #split into 12 attention heads
    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    #putting the pieces together
    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query) #12 query heads 
        key = self.split_heads(key, k=True) #12 hey heads 
        value = self.split_heads(value) #12 value heads
        if layer_past is not None: #add previous/past information 
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value) #perform attention 
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

#multi layer perceptron
class MLP(nn.Module):
    #set up 
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx) #expand to 3072 features
        self.c_proj = Conv1D(nx, n_state) #shrink back to 768
        self.act = gelu #use GELU as activation function. (how much should this matter )
    #actual performance
    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

#Normalize → Attention → Add original → Normalize → MLP → Add original → Output
#Block contains LayerNorm, Attention, Residual addition, LayerNorm, MLP, Residual addition

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

#whole stack, Embeddings, 12 blocks, final layernorm
class CoarseTransformerModel(nn.Module):
    def __init__(self, config):
        super(CoarseTransformerModel, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.total_vocab_size

        self.vis_embed_mat = nn.Linear(config.total_vocab_size, config.n_embd, bias=False)  #visit embeddimg 
        self.pos_embed_mat = nn.Embedding(config.n_positions, config.n_embd) #position embedding 
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    
    
    def forward(self, input_visits, position_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_visits.size(1) + past_length, dtype=torch.long,
                                        device=input_visits.device)
            position_ids = position_ids.unsqueeze(0).expand(input_visits.size(0), input_visits.size(1))

        inputs_embeds = self.vis_embed_mat(input_visits)
        position_embeds = self.pos_embed_mat(position_ids)
        hidden_states = inputs_embeds + position_embeds
        for block, layer_past in zip(self.h, past):
            hidden_states, _ = block(hidden_states, layer_past)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

#masked linear layer
class AutoregressiveLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.tril(torch.ones(in_features, out_features)).int())
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias) #multiply every weight by it's mask value

#code level
class FineAutoregressiveHead(nn.Module):
    def __init__(self, config):
        super(FineAutoregressiveHead, self).__init__()
        self.auto1 = AutoregressiveLinear(config.n_embd + config.total_vocab_size, config.n_embd + config.total_vocab_size)
        self.auto2 = AutoregressiveLinear(config.n_embd + config.total_vocab_size, config.n_embd + config.total_vocab_size)
        self.n_embd = config.n_embd
        self.tot_vocab = config.total_vocab_size
    
    #history comes from course transformer
    #used for training halo model
    def forward(self, history, input_visits):
        history = history[:,:-1,:]
        input_visits = input_visits[:,1:,:] #at each level model should have previous visit and current visit
        code_logits = self.auto2(torch.relu(self.auto1(torch.cat((history, input_visits), dim=2))))[:,:,self.n_embd-1:-1]
        return code_logits

    #used for generating new data
    def sample(self, history, input_visits):
        history = history[:,:-1,:]
        input_visits = input_visits[:,1:,:]
        currVisit = torch.cat((history, input_visits), dim=2)[:,-1,:].unsqueeze(1)
        code_logits = self.auto2(torch.relu(self.auto1(currVisit)))[:,:,self.n_embd-1:-1]
        return code_logits

class HALOModel(nn.Module):
    def __init__(self, config):
        super(HALOModel, self).__init__()
        self.transformer = CoarseTransformerModel(config) #coarse level 
        self.ehr_head = FineAutoregressiveHead(config) #code level
    
    #used for training
    def forward(self, input_visits, position_ids=None, ehr_labels=None, ehr_masks=None, past=None, pos_loss_weight=None):
        hidden_states = self.transformer(input_visits, position_ids, past)
        code_logits = self.ehr_head(hidden_states, input_visits)
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        #compare against real lvels
        if ehr_labels is not None:    
            shift_labels = ehr_labels[..., 1:, :].contiguous()
            loss_weights = None
            if pos_loss_weight is not None:
                loss_weights = torch.ones(code_probs.shape, device=code_probs.device)
                loss_weights = loss_weights + (pos_loss_weight-1) * shift_labels
            if ehr_masks is not None:
                code_probs = code_probs * ehr_masks
                shift_labels = shift_labels * ehr_masks
                if pos_loss_weight is not None:
                    loss_weights = loss_weights * ehr_masks

            bce = nn.BCELoss(weight=loss_weights)
            loss = bce(code_probs, shift_labels)
            return loss, code_probs, shift_labels
        
        return code_probs
    
    '''Loss note (happens at the end when code has already been produced)
       Binary cross entropy 
       if there are 1000 codes, HALO makes a prediction for the 1000 codes and calculates the loss for the 1000 codes'''
        #gets sent back through back propogation 
    
    #use trained model to generate codes, occurs for each visit
    '''"For each possible code, HALO predicts a probability, then we sample whether that code appears in the synthetic visit based on that probability."'''
    def sample(self, input_visits, random=True):
        sig = nn.Sigmoid()
        hidden_states = self.transformer(input_visits)
        i = 0
        while i < self.ehr_head.tot_vocab:
            next_logits = self.ehr_head.sample(hidden_states, input_visits)
            next_probs = sig(next_logits)
            if random:
                visit = torch.bernoulli(next_probs)
            else:
                visit = torch.round(next_probs)
            
            remaining_visit = visit[:,0,i:]
            nonzero = torch.nonzero(remaining_visit, as_tuple=True)[1]
            if nonzero.numel() == 0:
                break

            first_nonzero = nonzero.min()
            input_visits[:,-1,i + first_nonzero] = visit[:,0,i + first_nonzero]
            i = i + first_nonzero + 1
        
        return input_visits