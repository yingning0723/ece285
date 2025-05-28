import torch
from torch import nn

class ResidualConvBlock(nn.Module):
    """Convolutional Residual Block"""
    def __init__(self, in_channels, out_channels, is_res=False):
        super(ResidualConvBlock, self).__init__()
        self.is_res = is_res
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.handle_channel = None
        if self.is_res and not self.same_channels:
            self.handle_channel = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        
    def forward(self, x):
        x_out = self.conv2(self.conv1(x))
        if not self.is_res:
            return x_out
        else:
            if self.handle_channel:
                x_out = x_out + self.handle_channel(x)
            else:
                x_out = x_out + x
            return x_out / 1.414 # normalize

class UNetDown(nn.Module):
    """UNet downward block (i.e., contracting block)"""
    def __init__(self, in_channels, out_channels):
        super(UNetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels, True), # use skip-connection
                  ResidualConvBlock(out_channels, out_channels),
                  nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """UNet upward block (i.e., expanding block)"""
    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2), # double h & w sizes
            ResidualConvBlock(out_channels, out_channels, True), # use residual-connection
            ResidualConvBlock(out_channels, out_channels)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, x_skip):
        return self.model(torch.cat([x, x_skip], dim=1))


class EmbedFC(nn.Module):
    """Fully Connected Embedding Layer"""
    def __init__(self, input_dim, hidden_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
##########————————fine-graind————————##########
        #return self.model(x)[:, :, None, None]
        return self.model(x)
##########————————fine-graind————————##########

##########————————fine-graind————————##########
class FiLM(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, in_channels)
        self.beta = nn.Linear(emb_dim, in_channels)

    def forward(self, x, emb):
        gamma = self.gamma(emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(emb).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta



##########————————fine-graind————————##########

class ContextUnet(nn.Module):
    """Context UNet model
    Args:
        in_channels (int): Number of channels in the input image
        height (int): Height of the input image
        width (int): Width of the input image
        n_feat (int): Number of initial features i.e., hidden channels to which
            the input image be transformed
        n_cfeat (int): Number of context features i.e., class categories
        n_downs (int): Number of down (and up) blocks of UNet. Default: 2
    """
    def __init__(self, in_channels, height, width, n_feat, n_cfeat, n_downs=2):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.n_downs = n_downs

        ##########————————fine-graind————————##########
        self.unify_embs = nn.ModuleList([
            nn.Linear(2**i * n_feat, 2**i * n_feat)
            for i in range(n_downs, 0, -1)
        ])
        self.film_blocks = nn.ModuleList([
            FiLM(in_channels=2**i * n_feat, emb_dim=2**i * n_feat)
            for i in range(n_downs, 0, -1)
        ])
        ##########————————fine-graind————————##########

        # Define initial convolution
        self.init_conv = ResidualConvBlock(in_channels, n_feat, True)
        
        # Define downward unet blocks
        self.down_blocks = nn.ModuleList()
        for i in range(n_downs):
            self.down_blocks.append(UNetDown(2**i*n_feat, 2**(i+1)*n_feat))
        
        # Define at the center layers
        self.to_vec = nn.Sequential(
            nn.AvgPool2d((height//2**len(self.down_blocks), width//2**len(self.down_blocks))), 
            nn.GELU())
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(
                2**n_downs*n_feat, 
                2**n_downs*n_feat, 
                (height//2**len(self.down_blocks), width//2**len(self.down_blocks))),
            nn.GroupNorm(8, 2**n_downs*n_feat),
            nn.GELU()
        )
        
        # Define upward unet blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_downs, 0, -1):
            self.up_blocks.append(UNetUp(2**(i+1)*n_feat, 2**(i-1)*n_feat))

        # Define final convolutional layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.GELU(),
            nn.Conv2d(n_feat, in_channels, 1, 1)
        )

        # Define time & context embedding blocks 
        self.timeembs = nn.ModuleList([EmbedFC(1, 2**i*n_feat) for i in range(n_downs, 0, -1)])
        self.contextembs = nn.ModuleList([EmbedFC(n_cfeat, 2**i*n_feat) for i in range(n_downs, 0, -1)])

    def forward(self, x, t, c):
        x = self.init_conv(x)
        downs = []
        for i, down_block in enumerate(self.down_blocks):
            if i == 0: downs.append(down_block(x))
            else: downs.append(down_block(downs[-1]))
        up = self.up0(self.to_vec(downs[-1]))
        ##########————————fine-graind————————##########
        for i, (up_block, down, contextemb, timeemb, unify_emb, film_block) in enumerate(
            zip(self.up_blocks, reversed(downs), self.contextembs, self.timeembs,
                self.unify_embs, self.film_blocks)):
            
            t_feat = timeemb(t)
            if c is not None:
                context_feat = contextemb(c)
                combined_emb = context_feat + t_feat  # shape [batch, 2**i * n_feat]
                combined_emb = combined_emb.view(combined_emb.shape[0], -1)
                combined_emb = unify_emb(combined_emb)  # → [batch, 2**(i+1) * n_feat]

                assert combined_emb.shape[1] == up.shape[1], f"Mismatch at layer {i}: combined_emb {combined_emb.shape}, up {up.shape}"
                
                up = up_block(film_block(up, combined_emb), down)
            else:
                up = up_block(up + t_feat, down)
        ##########————————fine-graind————————##########
        ### below is correct
        # for up_block, down, contextemb, timeemb in zip(self.up_blocks, downs[::-1], self.contextembs, self.timeembs):
        #   context_factor = contextemb(c) if c is not None else 1
        #   up = up_block(up * context_factor + timeemb(t), down)
        return self.final_conv(torch.cat([up, x], axis=1))
