use crate::action::{ActionManager,ActionType,INVALID_CARD};

#[derive(Debug,Clone,Copy,Default)]
pub struct RuleConfig {
    pub eight_cut:bool,//8切り
    pub eleven_back:bool,//Jバック
    pub spade_3_beat:bool,//スぺ3返し
    pub skip_five:bool,//5飛ばし
}

#[derive(Debug,Clone,Default)]
pub struct HandEffects {
    pub eight_cut:bool,
    pub eleven_back:bool,
    pub skip_five_count:usize,
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

        // 5飛びの判定
        if self.config.skip_five {
            match info.action_type {
                ActionType::Group => {
                    let card_idx = info.required_cards[0] as usize;
                    if card_idx != INVALID_CARD as usize && card_idx < 52 {
                        // 1枚目が5なら全カード5なので、size分スキップ
                        if card_idx % 13 == 2 {
                            effects.skip_five_count = info.size as usize;
                        }
                    }
                }
            
                ActionType::Stair => {
                    // 階段の開始Rankと終了Rankの範囲内に「5 (Rank index: 2)」が含まれるかチェック
                    let start_rank = (info.required_cards[0] as usize) % 13;
                    let end_rank = start_rank + (info.size as usize) - 1;

                    // 範囲内（start <= 2 <= end）にあれば5は確実に1枚だけ存在する
                    if start_rank <= 2 && 2 <= end_rank {
                        effects.skip_five_count = 1;
                    }
                }
            
                _ => {}
            }
        }

        effects
    }
}