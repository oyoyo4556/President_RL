use crate::env::RawState;

pub const PASS_ACTION_ID:usize = 0;
pub const TRAIN_AGENT_ID: usize = 0;
pub const ACTION_SIZE:usize = 462;
pub const JOKER_SINGLE_ACTION_ID: usize = 460;
pub const JOKER_PAIR_ACTION_ID: usize = 461;
pub const INPUT_STATE_DIM: usize = 126;
pub const NUM_PLAYERS:usize = 4;
pub const AVE_RANK:f32 = (NUM_PLAYERS + 1) as f32 / (2f32) - 1f32;//1位が0だから

#[derive(Clone)]
pub struct Experience {
    pub state:RawState,
    pub action:u16,
    pub reward:f32,
    pub next_state:RawState,
    pub done:bool,
    pub next_gamma: f32,
}