use president::env::{DaifugoEnv};
use president::agent::{DQNAgent, Opponent,Agent};
use president::common::{NUM_PLAYERS, TRAIN_AGENT_ID};
use std::fs;
use std::path::Path;


fn main() {
    println!("========================================");
    println!("         大富豪 RL環境 Render テスト      ");
    println!("========================================");

    let save_dir ="checkpoints".to_string();
    if !Path::new(&save_dir).exists() {
        fs::create_dir_all(&save_dir).expect("Failed to create save directory.");
        println!("Created directory: {}",save_dir);
    }


    // 1. エージェントと対戦相手（Opponent）の初期化
    let agent_id = TRAIN_AGENT_ID; 
    let mut agent = DQNAgent::new(100,1);
    agent.load("checkpoints/dqn_v1.0.2_daifugo_ep100000.safetensors").expect("Failed to load model.check the path!");
    agent.epsilon = 0.0;
    
    let mut opp = DQNAgent::new(100,1);
    agent.copy_weights_to(&mut opp).expect("Failed weights_copy to opponent");
    opp.epsilon = 0.0;

    let opponent = Opponent::DQN(opp);
    let mut env = DaifugoEnv::new(agent_id, opponent);
    
    // 前回の順位がないと交換フェーズが走らないため、テスト用に前回の順位をモック
    // 0:大富豪, 1:富豪, 2:貧民, 3:大貧民 と仮定
    env.state.previous_rankings = vec![0, 1, 2, 3]; 

    // 環境の初期化 (カード分配 + 交換フラグのセット)
    // ゲーム開始
    let mut state = env.reset();
    let mut done = false;
    let mut total_step = 0;

    println!("==================================================");
    println!("◆ 大富豪 ゲーム開始 ◆");
    println!("初期手札:");
    for p in 0..4 {
        println!("  Player {}: {:?}", p, state.hand_to_strings(p));
    }
    println!("==================================================");

    while !done {
        // --- 【RENDER】行動選択の前に、現在の「選択肢（合法手）」を一覧表示 ---
        let p = state.current_player;
        
        println!("\n==================================================");
        if state.exchange_phase {
            println!("【カード交換フェーズ】 Player {}", p);
        } else {
            println!("     Trun {}",&total_step);
            println!("===============================================");
            println!("【メインプレイ (革命: {})】 Player {} の手番", state.is_revolution, p);
            println!("【あがりプレイヤー】 {:?}",state.finished_order);

            let mut hand_len = Vec::new();
            for i in 0..NUM_PLAYERS {
                hand_len.push(state.hands[i].count_ones());
            }
            println!("【相手の残り枚数】[p0,p1,p2,p3]:{:?}",hand_len);
            if let Some(field_act) = state.current_field_action {
                let field_info = &env.action_manager.infos[field_act as usize];
                println!("  [現在の場札]: {}", field_info.to_readable_string());
            } else {
                println!("  [現在の場札]: なし (親の手番)");
            }
        }
        println!("  [現在の手札]: {:?}", state.hand_to_strings(p));

        // 💡 合法手（maskがtrueのもの）をスキャンして表示
        println!("  [出せる手一覧]:");
        let mut legal_count = 0;
        for (act_id, &is_legal) in state.legal_actions_mask.iter().enumerate() {
            if is_legal {
                let legal_info = &env.action_manager.infos[act_id];
                println!("    - ID: {:3} => {}", act_id, legal_info.to_readable_string());
                legal_count += 1;
            }
        }
        if legal_count == 0 {
            println!("    (出すことができるカードがありません)");
        }
        println!("--------------------------------------------------");

        // エージェントの行動選択
        let action = agent
            .select_action(&state, state.current_player)
            .expect("Failed to select action during main turn");

        // --- 【RENDER】実際に選択された行動を表示 ---
        let act_info = &env.action_manager.infos[action as usize];
        println!("  ★ Player {} の実際の選択 => [ID: {:3}] {}", p, action, act_info.to_readable_string());
        println!("==================================================");

        if state.exchange_phase {
            (state, _, _) = env.exchange_step(action as usize);
            continue;
        }

        let (next_state, _, is_done) = env.step(action);

        state = next_state;
        done = is_done;
        total_step += 1;
    }

    println!("\n==================================================");
    println!("◆ ゲーム終了 ◆");
    println!("総ステップ数: {}", total_step);
    println!("あがり順 (Finished Order): {:?}", env.state.finished_order);
    println!("==================================================");
}