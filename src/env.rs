use crate::agent::{Opponent,Agent};
use crate::action::{ ActionManager, ActionType};
use rand::seq::SliceRandom;
use crate::card::Card;
use crate::common::{JOKER_SINGLE_ACTION_ID, PASS_ACTION_ID, JOKER_PAIR_ACTION_ID,NUM_PLAYERS};
use crate::rule::{RuleConfig,RuleEvaluator,HandEffects};

#[derive(Debug,Clone)]
pub struct RawState {
    pub hands:[u64;NUM_PLAYERS],
    pub current_field_action:Option<u16>,
    pub current_player:usize,
    pub finished_order:Vec<u8>,
    pub alive_players:u8,
    pub action_log:[u16;NUM_PLAYERS],
    pub legal_actions_mask:[bool;462],
    pub is_revolution:bool,
    pub is_parmanent_revolution:bool,
    pub field_owner:Option<usize>,
    pub passed_players:u8,
    pub previous_rankings:Vec<u8>,
    pub exchange_phase:bool,
}


#[derive(Debug,Clone)]
pub struct ExchangeBuffer {
    pub from:usize,
    pub to:usize,
    pub card:u8,
}

pub struct DaifugoEnv {
    pub agent_id:usize,
    pub opponent:Opponent,
    pub action_manager:ActionManager,
    pub state:RawState,
    pub exchange_buffer:Vec<ExchangeBuffer>,
    pub exchange_turn_idx:u8,
    pub evaluator:RuleEvaluator,
}

impl DaifugoEnv {
    pub fn new(agent_id:usize,opponent:Opponent,rule:RuleConfig) -> Self {

        Self {
            agent_id,
            opponent,
            action_manager:ActionManager::new(),
            state:RawState {
                hands:[0u64;NUM_PLAYERS],
                current_field_action:None,
                current_player:0,
                finished_order:Vec::new(),
                alive_players:0b1111,
                action_log:[0;NUM_PLAYERS],
                legal_actions_mask:[false;462],
                is_revolution:false,
                is_parmanent_revolution:false,
                field_owner:None,
                passed_players:0,
                previous_rankings:Vec::new(),
                exchange_phase:false,
            },
            exchange_buffer:Vec::new(),
            exchange_turn_idx:0,
            evaluator:RuleEvaluator::new(rule),
        }
    }

    pub fn reset(&mut self) -> RawState{

        self.state.hands = [0u64;NUM_PLAYERS];
        self.state.current_field_action = None;
        self.state.current_player = 0;
        self.state.finished_order.clear();
        self.state.alive_players = 0b1111;
        self.state.action_log = [0;NUM_PLAYERS];
        self.state.legal_actions_mask = [false;462];
        self.state.is_revolution = false;
        self.state.is_parmanent_revolution = false;
        self.state.field_owner = None;
        self.state.passed_players = 0;
        self.exchange_buffer.clear();
        self.deal_cards();
        if self.state.previous_rankings.is_empty() {
            self.state.exchange_phase = false;
        } else {
            self.state.exchange_phase =true;
            self.exchange_turn_idx = 0;
            self.state.current_player = self.state.previous_rankings[0] as usize;
        }
        self.update_exchange_legal_actions(self.agent_id);

        self.get_raw_state()
    }

    pub fn get_raw_state(&self) -> RawState {
        self.state.clone()
    }

    pub fn exchange_step(&mut self,action:usize) -> (RawState,f32,bool){

        let daifugo = self.state.previous_rankings[0] as usize;
        let fugo = self.state.previous_rankings[1] as usize;
        let rev_previous_rankings = self.state.previous_rankings.iter().rev().cloned().collect::<Vec<u8>>();
        let hinmin = rev_previous_rankings[1] as usize;//平民を定義しない代わりに後ろからの順番で定義(プレイ人数に依存しない)
        let daihinmin = rev_previous_rankings[0] as usize;

        match self.exchange_turn_idx {
            //大富豪1枚目
            0 => {
                self.exchange_select(daifugo,daihinmin,action);
                self.exchange_turn_idx  = 1;
                self.update_exchange_legal_actions(daifugo);
                self.state.current_player = daifugo;
            }

            //大富豪2枚目
            1 => {
                self.exchange_select(daifugo,daihinmin,action);
                self.exchange_turn_idx = 2;
                self.update_exchange_legal_actions(fugo);
                self.state.current_player = fugo;
            }

            //富豪以降の処理
            2 => {
                self.exchange_select(fugo,hinmin,action);
                
                self.exchange_strongest(hinmin,fugo,1);
                self.exchange_strongest(daihinmin,daifugo,2);

                self.commit_exchange();

                self.state.current_player = daihinmin;
                self.update_legal_actions();
                while self.state.current_player != self.agent_id{
                    self.opponent_turn().expect("Opponent failed during step");
                }
            }

            _ => {
                panic!("invalid exchange state")
            }
        }

        (self.get_raw_state(),0.0,false)
    }


    fn exchange_select(&mut self,from:usize,to:usize,agent_action:usize) {
        
            let action = if from == self.agent_id {
                agent_action as u16
            } else {
                self.opponent.select_action(&self.state,from).expect("Exchange Failed")
            };

            let info = &self.action_manager.infos[action as usize];
            let card = info.required_cards[0];

            //仮抜き
            self.state.hands[from] &= !(1u64 << card);

            self.exchange_buffer.push(
                ExchangeBuffer {from,to,card}
            );
    }

    fn exchange_strongest(&mut self,from:usize,to:usize,count:usize) {
        let cards = self.extract_strongest_cards(from,count);

        for card in cards {
            self.state.hands[from] &= !(1u64 << card);

            self.exchange_buffer.push(
                ExchangeBuffer { from, to, card }
            )
        }
    }

    fn commit_exchange(&mut self) {
        for ex in &self.exchange_buffer {
            self.state.hands[ex.to] |= 1u64 << ex.card;
        }
        self.exchange_buffer.clear();
        self.state.exchange_phase = false;
        self.exchange_turn_idx = 0;
    }

    fn deal_cards(&mut self) {

        let mut deck:Vec<u8> = (0..54).collect();

        let mut rng = rand::rng();

        deck.shuffle(&mut rng);

        for (i,&card) in deck.iter().enumerate() {

            let player = i % NUM_PLAYERS;

            self.state.hands[player] |= 1u64 << card;
        }
    }

    fn update_exchange_legal_actions(&mut self,player:usize) {

        self.state.legal_actions_mask.fill(false);

        let hand =self.state.hands[player];

        for info in &self.action_manager.infos {
            if info.id == 0 {
                continue;
            }

            if info.action_type != ActionType::Group {
                continue;
            }

            if info.size != 1 {
                continue;
            }

            let owned = hand & info.required_mask;

            if owned != info.required_mask {
                continue;
            }

            self.state.legal_actions_mask[info.id] = true;
        }
    }

    fn extract_strongest_cards(&self,player:usize,count:usize) -> Vec<u8> {
        let hand = self.state.hands[player];
        let mut result =Vec::with_capacity(count);

        for joker in [52,53] {
            if ((hand >> joker) & 1) == 1 {
                result.push(joker as u8);

                if result.len() == count {
                    return result;
                }
            }
        }

        for rank in (0..13).rev() {
            for suit in 0..4 {
                let card = suit * 13 + rank;
                if ((hand >> card) & 1) == 1 {
                    result.push(card as u8);
                    if result.len() == count {
                        return result;
                    }
                }
            }
        }
        result
    }

    pub fn step(&mut self,action:u16) -> (RawState,f32,bool){

        let player = self.state.current_player;
        let mut reward = 0.0;

        if !self.state.legal_actions_mask[action as usize] {
            panic!("illegal action");
        }

        let effects = self.evaluator.evaluate_effects(action as usize, &self.action_manager);

        self.apply_action(player,action);

        self.update_field(action, &effects);

        self.update_finish(player);

        let done = self.check_done();
        if done {
            self.finalize_last_player();
            let rewards = self.compute_reward_vector();
            reward += rewards[self.agent_id];
            let mut final_state = self.get_raw_state();
            final_state.legal_actions_mask.fill(false);
            return (final_state,reward,true);
        }

        self.resolve_next_turn(&effects,player);

        self.update_legal_actions();

        while self.state.current_player != self.agent_id{
            self.opponent_turn().expect("Opponent failed during step");
            if self.check_done() {
                self.finalize_last_player();
                let rewards = self.compute_reward_vector();
                reward += rewards[self.agent_id];
                let mut final_state = self.get_raw_state();
                final_state.legal_actions_mask.fill(false);
                return (final_state,reward,true);
            }
        }

        return(self.get_raw_state(),0.0,false);
    }

pub fn apply_action(&mut self, player: usize, action: u16) {
    // pass
    if action == PASS_ACTION_ID as u16 {
        self.state.passed_players |= 1 << player; 
        self.state.action_log[player] = PASS_ACTION_ID as u16;
        return;
    }

    let info = &self.action_manager.infos[action as usize];
    let mut hand = self.state.hands[player];

    // 💡 mut を使わず、if-else の結果を直接代入する（上書きを発生させない）
    let required_joker_count = if action == JOKER_SINGLE_ACTION_ID as u16 {
        1
    } else if action == JOKER_PAIR_ACTION_ID as u16 {
        2
    } else {
        // 通常カードのアクションの場合、足りない枚数分だけJokerが必要
        let owned = hand & info.required_mask;
        hand &= !owned; // 通常カード（リアルに持っているカード）を消費
        
        (info.required_mask & !owned).count_ones() // 足りないカード枚数
    };

    // 2. Jokerを指定枚数分、手札から消費する
    if required_joker_count > 0 {
        let mut removed = 0;
        
        // 52番目のJokerを持っていたら消費
        if ((hand >> 52) & 1) == 1 && removed < required_joker_count {
            hand &= !(1u64 << 52);
            removed += 1;
        }
        // 53番目のJokerを持っていたら消費
        if ((hand >> 53) & 1) == 1 && removed < required_joker_count {
            hand &= !(1u64 << 53);
            removed += 1;
        }

        if removed < required_joker_count {
            panic!("not enough jokers");
        }
    }

    // 3. 状態の更新
    self.state.hands[player] = hand;
    self.state.action_log[player] = action;

}

    fn update_field(&mut self,action:u16,effects:&HandEffects) {

        // 💡 完全に原因をあぶり出すためのデバッグログ
        //if action == PASS_ACTION_ID as u16 {
            //println!(
            //"[DEBUG PASS] player: {}, passed_mask: {:b}, alive_mask: {:b}",
            //self.state.current_player,
            //self.state.passed_players,
            //self.state.alive_players
            //);
        //}

        // pass
        if action == PASS_ACTION_ID as u16 {
            let alive_count =
                self.state.alive_players.count_ones();
            let pass_count =
                (self.state.passed_players & self.state.alive_players).count_ones();

            let owner_alive = if let Some(owner) = self.state.field_owner {
                ((self.state.alive_players >> owner) & 1) == 1
            } else {
                false
            };

            let required_passes = if owner_alive {
                alive_count - 1
            } else {
                alive_count
            };

            if pass_count >= required_passes {
                self.state.current_field_action = None;
                self.state.field_owner = None;
                self.state.passed_players = 0;
                self.state.is_revolution = self.state.is_parmanent_revolution;//場が流れたらJバックを戻す
            }
            return;
        }

        self.state.current_field_action =
            Some(action);

        self.state.field_owner =
            Some(self.state.current_player);

        self.state.passed_players = 0;

        let info = &self.action_manager.infos[action as usize];
        //4枚出しの革命->Jバック革命->8切りの順で処理しないと革命が正しく更新されません
        //4枚だしの革命の処理
        if info.is_revolution_trigger {
            self.state.is_parmanent_revolution  = !self.state.is_parmanent_revolution;
            self.state.is_revolution = self.state.is_parmanent_revolution;
        }

        //Jバックの処理
        if effects.eleven_back {
            self.state.is_revolution = !self.state.is_revolution;
        }
        //8切りの処理
        if effects.eight_cut {
            self.state.current_field_action = None;
            self.state.field_owner = None;
            self.state.passed_players = 0;
            self.state.is_revolution = self.state.is_parmanent_revolution;//場が流れたらJバックを戻す
            return;
        }

    }

    fn update_finish(&mut self,player:usize) {
        if self.state.hands[player] != 0 {
            return;
        }

        if self.state.finished_order.contains(&(player as u8)) {
            return;
        }

        self.state.finished_order.push(player as u8);

        self.state.alive_players &= !(1u8 << player);
        self.state.action_log[player] = PASS_ACTION_ID as u16;//未行動はpassとしてAgentに認識してもらう
    }

    fn check_done(&self) -> bool {
        self.state.alive_players.count_ones() <= 1
    }

    fn advance_player(&mut self) {

        for _ in 0..NUM_PLAYERS {
            self.state.current_player =
                (self.state.current_player + 1)% NUM_PLAYERS;

            let alive =
                (self.state.alive_players >> self.state.current_player) & 1;

            if alive == 1 {
                return;
            }
        }

        panic!("No alive players");
    }

    pub fn resolve_next_turn(&mut self,effects:&HandEffects,current_player:usize) {
        let player_cleared = ((self.state.alive_players >> current_player) & 1) == 0;

        if effects.eight_cut && !player_cleared {
            return;
        }

        self.advance_player();
    }

    fn update_legal_actions(&mut self) {

        self.state.legal_actions_mask =
            [false;462];

        let player =
            self.state.current_player;

        let hand =
            self.state.hands[player];

        let joker_count =
            ((hand >> 52) & 1)
            +
            ((hand >> 53) & 1);

        if self.state.current_field_action.is_some() {
            self.state.legal_actions_mask[0] = true;
        }

        for info in &self.action_manager.infos {

            // pass always legal
            if info.id == 0 {
                continue;
            }

            if info.id == JOKER_SINGLE_ACTION_ID {
                if joker_count >= 1 {
                    if let Some(field_action) = self.state.current_field_action {
                        let field_info = &self.action_manager.infos[field_action as usize];
                        if self.can_beat(info,field_info) {
                            self.state.legal_actions_mask[info.id] = true;
                        }
                    }
                }
                continue;
            }

            if info.id == JOKER_PAIR_ACTION_ID {
                if joker_count >= 2 {
                    if let Some(field_action) = self.state.current_field_action {
                        let field_info = &self.action_manager.infos[field_action as usize];
                        if self.can_beat(info,field_info)  {
                            self.state.legal_actions_mask[info.id] = true;
                        }
                    }
                }
                continue;
            }

            if self.evaluator.config.spade_3_beat && info.required_mask == 1u64 << 0 {
                if (hand & (1u64 << 0)) == 0 {
                    continue;
                }
            }

            let missing =
                info.required_mask & !hand;

            if missing.count_ones() > joker_count as u32
            {continue;}

            let owned_real_cards = hand &info.required_mask;
            if owned_real_cards.count_ones() == 0 {
                continue;
            }

            // field check
            if let Some(field_action) =
                self.state.current_field_action {

                let field_info =
                    &self.action_manager
                        .infos[field_action as usize];

                if !self.can_beat(info,field_info) {continue;}
            }
            self.state.legal_actions_mask[info.id]= true;
        }
    }

    fn can_beat(
        &self,
        action:&crate::action::ActionInfo,
        field:&crate::action::ActionInfo,
    ) -> bool {

        if action.size != field.size {
            return false;
        }

        if action.action_type != field.action_type {
            return false;
        }

        if action.action_type == ActionType::Stair 
           && action.size != field.size {
            return false;
        }

        //♠3返し
        //先にこっちを判定することでjokerにjokerで返せるバグを防いでいる
        if field.id == JOKER_SINGLE_ACTION_ID {
            if self.evaluator.config.spade_3_beat && action.required_mask == 1u64  << 0{
                return true;
            }
            return false;
        }

        //革命時でもジョーカーは最強
        if action.id == JOKER_SINGLE_ACTION_ID {
            return true;
        }

        if action.id == JOKER_PAIR_ACTION_ID {
            return true;
        }

        if !self.state.is_revolution {
            action.strength > field.strength
        } else {
            action.strength < field.strength
        }
    }

    fn opponent_turn(&mut self) -> Result<(),String> {
        let player = self.state.current_player;

        let action = self.opponent.select_action(&self.state,player)?;
        let effects = self.evaluator.evaluate_effects(action as usize, &self.action_manager);

        self.apply_action(player, action);
        self.update_field(action, &effects);
        self.update_finish(player);
        self.resolve_next_turn(&effects, player);
        self.update_legal_actions();

        Ok(())
    }

    fn finalize_last_player(&mut self) {
        for player in 0..NUM_PLAYERS {
            let alive = ( self.state.alive_players >> player) & 1;
            if alive == 1 {
                self.state.finished_order.push(player as u8);
                self.state.alive_players = 0;
                break;
            }
        }

        self.state.previous_rankings = self.state.finished_order.clone();
    }

    fn compute_reward_vector(&self) -> Vec<f32> {
        let mut rewards = vec![0.0;NUM_PLAYERS];
        let values = [1.0,0.3,-0.3,-1.0];//差をつけてるのはわざと
        for (i,&p_idx) in self.state.finished_order.iter().enumerate(){
            rewards[p_idx as usize] = values[i];
        }

        rewards
    }
}

impl RawState {
    /// プレイヤーの手札を `["♠3", "♣5", "Joker(1)"]` のような文字列のベクタにする
    pub fn hand_to_strings(&self, player: usize) -> Vec<String> {
        let hand = self.hands[player];
        let mut card_strs = Vec::new();

        // 通常カードを順番（インデックス順）にチェック
        for idx in 0..52 {
            if ((hand >> idx) & 1) == 1 {
                card_strs.push(Card::from_index(idx).to_display_string());
            }
        }
        // ジョーカーのチェック
        if ((hand >> 52) & 1) == 1 { card_strs.push("Joker(1)".to_string()); }
        if ((hand >> 53) & 1) == 1 { card_strs.push("Joker(2)".to_string()); }

        card_strs
    }

    pub fn used_card_to_string(&self) -> Vec<String> {
        let mut used_card = (1u64 << 54) - 1;
        let mut used_card_strs = Vec::new();
            for i in 0..NUM_PLAYERS {
                used_card &= !self.hands[i];
            }

            for idx in 0..52{
                if ((used_card >> idx) & 1) == 1 {
                    used_card_strs.push(Card::from_index(idx).to_display_string());
                }
            }

            if ((used_card >> 52) & 1) == 1 {used_card_strs.push("Joker(1)".to_string());}
            if ((used_card >> 53) & 1) == 1 {used_card_strs.push("Joker(2)".to_string());}

            used_card_strs
    }
}
