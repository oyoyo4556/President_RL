use crate::action::{ActionManager,ActionType,INVALID_CARD};

#[derive(Debug,Clone,Copy,Default)]
pub struct RuleConfig {
    pub eight_cut:bool,//8切り
    pub eleven_back:bool,//Jバック
    pub spade_3_beat:bool,//スぺ3返し
}

#[derive(Debug,Clone,Default)]
pub struct HandEffects {
    pub eight_cut:bool,
    pub eleven_back:bool,
}

pub struct RuleEvaluator {
    pub config:RuleConfig,
}

impl RuleEvaluator {
    pub fn new(config:RuleConfig) -> Self {
        Self {config}
    }

    pub fn evaluate_effects(&self,action_id:usize,action_manager:&ActionManager) -> HandEffects {
        let mut effects = HandEffects::default();
        let info = &action_manager.infos[action_id];

        //8切りの判定
        if self.config.eight_cut && info.action_type == ActionType::Group {
            let card_idx = info.required_cards[0] as usize;
            if card_idx != INVALID_CARD as usize && card_idx < 52 {
                if card_idx % 13 == 5 {
                    effects.eight_cut = true;
                }
            }
        }

        //Jバックの判定
        if self.config.eleven_back && info.action_type == ActionType::Group {
            let card_idx = info.required_cards[0] as usize;
            if card_idx != INVALID_CARD as usize && card_idx < 52 {
                if card_idx % 13 == 8 {
                    effects.eleven_back = true;
                }
            }
        }

        effects
    }
}