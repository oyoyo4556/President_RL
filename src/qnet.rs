use candle_core::{Result,Tensor};
use candle_nn::{linear,Linear,Module,VarBuilder,LayerNorm,LayerNormConfig};

pub struct ResidualBlock{
    fc1:Linear,
    ln1:LayerNorm,
    fc2:Linear,
}

impl ResidualBlock {
    pub fn new(dim:usize,vb:VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(dim,2*dim,vb.pp("fc1"))?;
        let ln1 = candle_nn::layer_norm(dim,LayerNormConfig::default(),vb.pp("ln1"))?;
        let fc2 = candle_nn::linear(dim*2,dim,vb.pp("fc2"))?;
        Ok(Self{fc1,ln1,fc2})
    }

    pub fn forward(&self,x:&Tensor) -> Result<Tensor> {
        let residual = x;

        let mut out = self.ln1.forward(x)?;
        out = self.fc1.forward(&out)?;
        out = out.relu()?;
        out = self.fc2.forward(&out)?;

        out.add(residual)
    }

}

pub struct DuelingQNet{
    input_layer:Linear,
    res:ResidualBlock,
    final_ln:LayerNorm,

    value_buffer:Linear,
    value:Linear,

    advantage_buffer:Linear,
    advantage:Linear,
}

impl DuelingQNet{
    pub fn new(state_dim:usize,hidden_dim:usize,action_dim:usize,vb: VarBuilder) -> Result<Self> {

        let input_layer = linear(state_dim,hidden_dim,vb.pp("input_layer"))?;
        let res = ResidualBlock::new(hidden_dim,vb.pp("res"))?;
        let final_ln = candle_nn::layer_norm(hidden_dim,candle_nn::LayerNormConfig::default(),vb.pp("final_ln"))?;
        let value_buffer = linear(hidden_dim,64,vb.pp("value_buffer"))?;
        let value = linear(64,1,vb.pp("value"))?;
        let advantage_buffer = linear(hidden_dim,hidden_dim,vb.pp("advantage_buffer"))?;
        let advantage = linear(hidden_dim,action_dim,vb.pp("advantage"))?;

        Ok(Self {input_layer,res,final_ln,value_buffer,value,advantage_buffer,advantage})
    }

    pub fn forward(&self,x:&Tensor,mask:&Tensor) -> Result<Tensor> {
        let mut x = self.input_layer.forward(x)?;
        x = x.relu()?;
        x = self.res.forward(&x)?;
        x = self.final_ln.forward(&x)?;
        
        //value
        let mut v = self.value_buffer.forward(&x)?;
        v = v.relu()?;
        v = self.value.forward(&v)?;

        //advantage
        let mut a = self.advantage_buffer.forward(&x)?;
        a = a.relu()?;
        a = self.advantage.forward(&a)?;

        let masked_a = a.broadcast_mul(mask)?;
        let legal_counts = mask.sum_keepdim(1)?.affine(1.0,1e-8)?;
        
        let a_mean = masked_a.sum_keepdim(1)?.broadcast_div(&legal_counts)?;

        let advantage_centered = a.broadcast_sub(&a_mean)?.broadcast_mul(mask)?;
        let q = advantage_centered.broadcast_add(&v)?;

        Ok(q)
    }
}