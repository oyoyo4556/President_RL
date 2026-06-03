use crate::env::{RawState};
use std::cell::RefCell;
use candle_core::{Device,Result,Tensor};
use crate::common::{Experience,INPUT_STATE_DIM,ACTION_SIZE,NUM_PLAYERS,AVE_RANK};

pub struct Processor{
    states_buf:RefCell<Vec<f32>>,
    next_states_buf:RefCell<Vec<f32>>,
    masks_buf:RefCell<Vec<f32>>,
    next_masks_buf:RefCell<Vec<f32>>,
    pub infer_buf:RefCell<Vec<f32>>,//agentのselect_actionで使用
    pub infer_mask_buf:RefCell<Vec<f32>>,
}

impl Processor {

    pub fn new(max_batch_size:usize) -> Self {
        Self {
            states_buf:RefCell::new(Vec::with_capacity(max_batch_size*INPUT_STATE_DIM)),
            next_states_buf:RefCell::new(Vec::with_capacity(max_batch_size*INPUT_STATE_DIM)),
            masks_buf:RefCell::new(Vec::with_capacity(max_batch_size*462)),
            next_masks_buf:RefCell::new(Vec::with_capacity(max_batch_size*462)),
            infer_buf:RefCell::new(Vec::with_capacity(INPUT_STATE_DIM)),
            infer_mask_buf:RefCell::new(Vec::with_capacity(462))
        }
    }

    pub fn batch_to_tensors(&self,exps:&[&Experience],device:&Device,player_id:usize) 
    -> Result<(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor)> {

        let batch_size = exps.len();

        let mut s_buf = self.states_buf.borrow_mut();
        let mut ns_buf = self.next_states_buf.borrow_mut();
        let mut m_buf = self.masks_buf.borrow_mut();
        let mut nm_buf = self.next_masks_buf.borrow_mut();
        let mut actions_raw:Vec<u32> = Vec::with_capacity(batch_size);
        let mut rewards_raw:Vec<f32> = Vec::with_capacity(batch_size);
        let mut dones_raw:Vec<f32> = Vec::with_capacity(batch_size);
        let mut next_gammas_raw:Vec<f32> = Vec::with_capacity(batch_size);

        s_buf.clear();
        ns_buf.clear();
        m_buf.clear();
        nm_buf.clear();



        for exp in exps {
            self.write_buf(&mut s_buf,&exp.state,player_id);
            self.write_buf(&mut ns_buf,&exp.next_state,player_id);
            m_buf.extend(exp.state.legal_actions_mask.iter().map(|&b| if  b{1.0f32}else{0.0f32}));
            nm_buf.extend(exp.next_state.legal_actions_mask.iter().map(|&b| if b{1.0f32}else{0.0f32}));
            actions_raw.push(exp.action as u32);
            rewards_raw.push(exp.reward);
            dones_raw.push(if exp.done{1.0f32} else{0.0f32});
            next_gammas_raw.push(exp.next_gamma);
        }

        // 1. Candle(from_slice)がサボっている要素数チェックを、手前で厳密に実行する
        //candle_coreの0.10.2段階ではfrom_sliceは(usize,usize)が渡されると穴が無いので要素数チェックされません(Into<Shape>の一括実装が原因)
        //つまりINPUT_STATE_DIMはどんな値でも通り、境界外読み込みによりデータが捏造されます
        let required_elements = batch_size * INPUT_STATE_DIM;
        let required_mask_elements = batch_size * 462; 

        assert_eq!(
            s_buf.len(), 
            required_elements, 
            "【致命的バグ防止】s_bufの要素数 ({}) が、要求されたTensorのサイズ ({}) と一致しません！", 
            s_buf.len(), 
            required_elements
        );

        assert_eq!(
            ns_buf.len(), 
            required_elements, 
            "【致命的バグ防止】ns_bufの要素数({})が、要求されたTensorのサイズ({})と一致しません!",
            ns_buf.len(),
            required_elements
        );

        assert_eq!(m_buf.len(), required_mask_elements, "【致命的バグ防止】m_bufの要素数が一致しません!");
        assert_eq!(nm_buf.len(), required_mask_elements, "【致命的バグ防止】nm_bufの要素数が一致しません!");


        let states = Tensor::from_slice(&s_buf.as_slice(),(batch_size,INPUT_STATE_DIM),device)?;
        let next_states = Tensor::from_slice(&ns_buf.as_slice(),(batch_size,INPUT_STATE_DIM),device)?;
        let masks = Tensor::from_slice(&m_buf.as_slice(),(batch_size,462),device)?;
        let next_masks = Tensor::from_slice(&nm_buf.as_slice(),(batch_size,462),device)?;
        let actions = Tensor::from_vec(actions_raw,batch_size,device)?;
        let rewards = Tensor::from_vec(rewards_raw,batch_size,device)?;
        let dones = Tensor::from_vec(dones_raw,batch_size,device)?;
        let next_gammas = Tensor::from_vec(next_gammas_raw,batch_size,device)?;

        Ok((states,next_states,masks,next_masks,actions,rewards,dones,next_gammas))

    }


    //ここでstateの定義をする
    pub fn write_buf(&self,obs:&mut Vec<f32>,state:&RawState,player_id:usize) {

        //自分の手札情報[54]
        let my_hand = state.hands[player_id];
        for card in 0..54 {
            obs.push(if ((my_hand >> card) & 1) == 1 {1.0} else {0.0});
        }

        //使用されたカード[54]
        let mut used_card = (1u64 << 54) - 1;

        for i in 0..NUM_PLAYERS {
            let p_idx = (player_id + i) % NUM_PLAYERS;
            let hand_count = state.hands[p_idx].count_ones() as f32;
            let norm_act_log = state.action_log[p_idx] as f32 /(ACTION_SIZE -1) as f32;
            used_card &= !state.hands[p_idx];

            if state.previous_rankings.is_empty() {
                obs.push(AVE_RANK);
            } else {
                //自分と他プレイヤーの前回の結果をplyer_idから順番に並び替え。もしかしたら席順で戦略が変わるかも？[4]
                let mut rank_found = AVE_RANK;
                for j in 0..NUM_PLAYERS {
                    if state.previous_rankings[j] == p_idx as u8 {
                        rank_found = j as f32;
                        break;
                    }
                }
                obs.push(rank_found);
            }
            obs.push(hand_count/14.0);//対戦相手の残り手札枚数[NUM_PLAYERS](ここでは4)
            obs.push(norm_act_log)//1手分の履歴(相手の行動をみるため)[4]

        }

        for card in 0..54 {
            obs.push(if ((used_card >> card) & 1) == 1 { 1.0 } else { 0.0 });
        }

        //Jバックのフラグ[1]
        obs.push(if state.is_revolution{1.0}else{0.0});
        //革命フラグ[1]
        obs.push(if state.is_parmanent_revolution{1.0}else{0.0});

        //現在のfield_action_id[1](改善の余地あり)
        if let Some(act) = state.current_field_action {
            let norm_act = act as f32 / (ACTION_SIZE-1) as f32 ;
            obs.push(norm_act);
            
        } else {
            obs.push(-1.0);
        }

        //誰からこの場が始まったか[1]
        if let Some(field_owner) = state.field_owner {
            let norm_field_owner = field_owner as f32 / (NUM_PLAYERS -1) as f32;
            obs.push(norm_field_owner); 
        } else {
            //fieldが流れて誰も出してないときは-1
            obs.push(-1f32);
        }

        //この場でパスしたプレイヤーの数[1]
        let norm_passed_player = state.passed_players.count_ones() as f32 / (NUM_PLAYERS -1) as f32;
        obs.push(norm_passed_player);

        //残り人数[1]
        let norm_alive = state.alive_players.count_ones() as f32/NUM_PLAYERS as f32;
        obs.push(norm_alive);

        //交換フェーズかどうか[1]
        obs.push(if state.exchange_phase {1.0}else{0.0});

    }
}
