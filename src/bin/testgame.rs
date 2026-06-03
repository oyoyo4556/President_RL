use president::env::DaifugoEnv;
use president::agent::{Agent, Opponent, RandomAgent};
use president::card::{Card,Rank,Suit};
use president::rule::RuleConfig;

fn card_to_string(card:u8) -> String {

    match Card::from_index(card as usize) {

        Card::Joker1 => "J1".to_string(),
        Card::Joker2 => "J2".to_string(),

        Card::Normal { suit, rank } => {

            let suit_str = match suit {
                Suit::Spade => "♠",
                Suit::Club => "♣",
                Suit::Diamond => "♦",
                Suit::Heart => "♥",
            };

            let rank_str = match rank {
                Rank::Three => "3",
                Rank::Four => "4",
                Rank::Five => "5",
                Rank::Six => "6",
                Rank::Seven => "7",
                Rank::Eight => "8",
                Rank::Nine => "9",
                Rank::Ten => "10",
                Rank::Jack => "J",
                Rank::Queen => "Q",
                Rank::King => "K",
                Rank::Ace => "A",
                Rank::Two => "2",
            };

            format!("{}{}", suit_str, rank_str)
        }
    }
}

fn print_hand(hand:u64) {

    let mut cards = Vec::new();

    for i in 0..54 {

        if ((hand >> i) & 1) == 1 {
            cards.push(card_to_string(i as u8));
        }
    }

    println!("{:?}", cards);
}

fn print_legal_actions(env:&DaifugoEnv) {

    println!("legal actions:");

    for (i,&v) in
        env.state
            .legal_actions_mask
            .iter()
            .enumerate()
    {
        if !v {
            continue;
        }

        let info =
            &env.action_manager.infos[i];

        print!(
            "[{}: {:?} size={} str={}] ",
            i,
            info.action_type,
            info.size,
            info.strength,
        );
    }

    println!();
}

fn main() {

    let opponent =
        Opponent::Random(
            RandomAgent::new()
        );

    let rule = RuleConfig {
        eight_cut:true,
        eleven_back:true,
    };

    let mut env =
        DaifugoEnv::new(
            0,
            opponent,
            rule,
        );

    let agent = RandomAgent::new();

    for episode in 0..=2 {
        println!("\n=== EPISODE {} ===", episode);
        
    
        let mut state =
        env.reset();
        if episode == 2 {
            println!("previous_rankings:{:?}",env.state.previous_rankings);
        }

        println!("=== GAME START ===");


        loop {

            println!("\n====================");
            println!(
            "player: {}",
            state.current_player
            );

            println!(
            "exchange_phase: {}",
            state.exchange_phase
            );

            println!(
            "revolution: {}",
            state.is_revolution
            );

            println!(
            "field: {:?}",
            state.current_field_action
            );

            println!("alive players: {:?}",state.alive_players);

            println!("action_log:{:?}",state.action_log);

            println!("\nhand:");
            print_hand(
            state.hands[state.current_player]
            );

            println!();
            print_legal_actions(&env);

            let action =
                agent
                .select_action(
                    &state,
                    state.current_player,
                )
                .expect("Failed to select action during main turn");

            println!(
            "\naction selected: {}",
            action
            );

            println!("action_info: {:?}", env.action_manager.infos[action as usize]);

            if state.exchange_phase {

                (state,_,_) =
                    env.exchange_step(
                       action as usize
                    );

                println!("Exchange_buffer:{:?}",env.exchange_buffer);
                continue;
            }

            let (next_state,reward,done) =
                env.step(action);

            state = next_state;

            if done {

                println!("\n=== GAME END ===");

                println!(
                "finished order: {:?}",
                state.finished_order
                );

                println!(
                "reward: {}",
                reward
                );

                break;
            }
        }
    }
}