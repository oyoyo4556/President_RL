
use crate::common::{ACTION_SIZE,PASS_ACTION_ID,JOKER_SINGLE_ACTION_ID,JOKER_PAIR_ACTION_ID};
use crate::card::Card;

pub const INVALID_CARD:u8 = 255;

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum ActionType {
    Pass,
    Group,//重ね数字
    Stair,//階段
}

#[derive(Debug, Clone)]
pub struct ActionInfo {
    pub id: usize,
    pub action_type: ActionType,
    // action template cards
    pub required_cards: [u8;13],
    // bit representation
    pub required_mask: u64,
    // number of required cards
    pub size:u8,
    // base strength
    pub strength:i8,
    pub is_revolution_trigger: bool,
}

impl ActionInfo {
    pub fn to_readable_string(&self) -> String {
        match self.action_type {
            ActionType::Pass => "パス".to_string(),
            ActionType::Group | ActionType::Stair => {
                let mut card_strs = Vec::new();
                
                for i in 0..(self.size as usize) {
                    let card_idx = self.required_cards[i];
                    if card_idx != INVALID_CARD {
                        let card = Card::from_index(card_idx as usize);
                        card_strs.push(card.to_display_string());
                    }
                }

                let type_str = match self.action_type {
                    ActionType::Group => match self.size {
                        1 => "単発",
                        2 => "ペア",
                        3 => "3枚出し",
                        4 => "4枚出し (革命)",
                        _ => "グループ",
                    },
                    ActionType::Stair => "階段",
                    _ => unreachable!(),
                };

                let rev_str = if self.is_revolution_trigger { " ★革命トリガー" } else { "" };

                format!("{:?} {} (強さ: {}){}", card_strs, type_str, self.strength, rev_str)
            }
        }
    }
}



pub struct ActionManager {
    pub infos: Vec<ActionInfo>,
}

impl ActionManager {

    pub fn new() -> Self {

        // stair templates
        let mut stair_patterns = Vec::new();

        for length in 3..=13 {
            for start in 0..=(13-length) {
                stair_patterns.push((start,length));
            }
        }

        let mut infos = Vec::with_capacity(ACTION_SIZE);

        for action_id in 0..ACTION_SIZE {

            let (
                action_type,
                required_cards,
                size,
                strength,
                is_revolution_trigger,
            ) = Self::decode_action(action_id,&stair_patterns);

            let mut required_mask = 0u64;

            for i in 0..size {
                required_mask |= 1u64 << required_cards[i];
            }

            infos.push(ActionInfo {
                id:action_id,
                action_type,
                required_cards,
                required_mask,
                size:size as u8,
                strength:strength as i8,
                is_revolution_trigger,
            });
        }

        Self {
            infos,
        }
    }

    fn decode_action(
        action_id:usize,
        stair_patterns:&Vec<(usize,usize)>,
    ) -> (
        ActionType,
        [u8;13],
        usize,
        i32,
        bool,
    ) {

        let mut cards = [INVALID_CARD;13];
        let mut size = 0;

        // PASS
        if action_id == PASS_ACTION_ID {
            return (
                ActionType::Pass,
                cards,
                0,
                -1,
                false,
            );
        }

        // GROUP
        if (1..=195).contains(&action_id) {

            let idx = action_id - 1;

            let rank = idx / 15;

            let combo = (idx % 15) + 1;

            if combo & 1 != 0 {
                cards[size] = rank as u8;
                size += 1;
            }

            if combo & 2 != 0 {
                cards[size] = (13 + rank) as u8;
                size += 1;
            }

            if combo & 4 != 0 {
                cards[size] = (26 + rank) as u8;
                size += 1;
            }

            if combo & 8 != 0 {
                cards[size] = (39 + rank) as u8;
                size += 1;
            }

            return (
                ActionType::Group,
                cards,
                size,
                rank as i32,
                size == 4,
            );
        }

        // STAIR
        if (196..=459).contains(&action_id) {

            let idx = action_id - 196;

            let suit = idx / 66;

            let pattern_idx = idx % 66;

            let (start,length) = stair_patterns[pattern_idx];

            for i in 0..length {
                cards[size] =
                    (suit * 13 + start + i) as u8;

                size += 1;
            }

            return (
                ActionType::Stair,
                cards,
                size,
                (start + length - 1) as i32,
                length >= 4,
            );
        }

        // Joker single
        if action_id == JOKER_SINGLE_ACTION_ID {

            cards[0] = 52;

            return (
                ActionType::Group,
                cards,
                1,
                13,
                false,
            );
        }

        // Joker pair
        if action_id == JOKER_PAIR_ACTION_ID {

            cards[0] = 52;
            cards[1] = 53;

            return (
                ActionType::Group,
                cards,
                2,
                13,
                false,
            );
        }

        panic!("Invalid action_id: {}",action_id);
    }
}