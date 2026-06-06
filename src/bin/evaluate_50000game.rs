use president::env::DaifugoEnv;
use president::agent::{Agent, Opponent, RandomAgent};
use president::rule::RuleConfig;

fn main() {

    let rule = RuleConfig {
        eight_cut:true,
        eleven_back:true,
        spade_3_beat:true,
    };

    let opponent =
        Opponent::Random(
            RandomAgent::new()
        );

    let mut env =
        DaifugoEnv::new(
            0,
            opponent,
            rule,
        );

    let agent = RandomAgent::new();

    let num_episodes = 50000;
    let mut total_step = 0;

    let mut agent_ranks = Vec::new();
    let mut rank_counts = vec![0;4];
    env.opponent = Opponent::Random(RandomAgent::new());
    println!("=============================================================");
    println!("Starting evaluation vs RandomAgent for 50000 episodes");
    println!("=============================================================");

    for episode in 1..=num_episodes {
    
        let mut state =
        env.reset();

        let mut done = false;
        


        while !done {


            let action =
                agent
                .select_action(
                    &state,
                    state.current_player,
                )
                .expect("Failed to select action during main turn");

            if state.exchange_phase {

                (state,_,_) =
                    env.exchange_step(
                       action as usize
                    );

                continue;
            }

            let (next_state,_,is_done) =
                env.step(action);

            state = next_state;
            done = is_done;
            total_step += 1;

        }

        let final_ranks = env.state.finished_order.clone();
        let agent_rank = final_ranks.iter().position(|&p| p == 0).expect("Failed to find agent's rank");
        agent_ranks.push(agent_rank);
        rank_counts[agent_rank] += 1;

        if episode % 5000 == 0 {
                let r1_rate = rank_counts[0] as f32/episode as f32 *100.0;
                let r2_rate = rank_counts[1] as f32/episode as f32 *100.0;
                let r3_rate = rank_counts[2] as f32/episode as f32 *100.0;
                let r4_rate = rank_counts[3] as f32/episode as f32 *100.0;
                let ave_rank = agent_ranks.iter().sum::<usize>() as f32 / agent_ranks.len() as f32 + 1.0;
                println!("Games:{:>5} | 1st:{:>5.2}% | 2nd:{:>5.2}% | 3rd:{:>5.2}% | 4th:{:>5.2}% | Ave_Rank:{:>5.2}",
                episode,r1_rate,r2_rate,r3_rate,r4_rate,ave_rank);
        }
        
    }

    let r1_rate = rank_counts[0] as f32/num_episodes as f32 *100.0;
    let r2_rate = rank_counts[1] as f32/num_episodes as f32 *100.0;
    let r3_rate = rank_counts[2] as f32/num_episodes as f32 *100.0;
    let r4_rate = rank_counts[3] as f32/num_episodes as f32 *100.0;
    let ave_rank = agent_ranks.iter().sum::<usize>() as f32 / agent_ranks.len() as f32 + 1.0;
    println!("========================================================");
    println!("Final Result {} Games vs RandomAgent",num_episodes);
    println!("Mainagent Win Rate (1st place): {:.2}%",rank_counts[0] as f32/num_episodes as f32 *100.0);
    println!("Rank Rate: 1st:{:.2}% | 2nd:{:.2}% | 3rd:{:.2}% | 4th:{:.2}%",r1_rate,r2_rate,r3_rate,r4_rate);
    println!("Ave_Rank : {:.4}",ave_rank);
    println!("Total_Steps : {}",total_step);
    println!("========================================================");

}