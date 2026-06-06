use std::fs;
use std::path::Path;
use president::env::{DaifugoEnv};
use president::agent::{RandomAgent,DQNAgent,Opponent};
use president::trainer::Trainer;
use president::rule::RuleConfig;
use president::common::INPUT_STATE_DIM;

fn main(){
    let save_dir ="checkpoints".to_string();
    if !Path::new(&save_dir).exists() {
        fs::create_dir_all(&save_dir).expect("Failed to create save directory.");
        println!("Created directory: {}",save_dir);
    }

    let eta_max = 1e-4;
    let eta_min = 1e-5;
    let t_0 = 6000;
    let t_mult = 2;

    let batch_size = 64;
    let tau = 0.005;
    let save_interval = 10000;
    let num_episodes = 100_000;
    let agent_name = "dqn_v1.1.0_8J3".to_string();

    let mut agent = DQNAgent::new(100_000,3);
    let opp_agent = RandomAgent::new();
    //agent.copy_weights_to(&mut opp_agent).expect("failed copy_weight to opponent!");
    //opp_agent.epsilon = 0.0;
    let rule = RuleConfig {
        eight_cut:true,
        eleven_back:true,
        spade_3_beat:true,
    };
    let opponent = Opponent::Random(opp_agent);
    let mut env = DaifugoEnv::new(0,opponent,rule);
    let mut trainer = Trainer::new(
        eta_max,
        eta_min,
        t_0,
        t_mult,
        batch_size,
        tau,
        save_dir,
        save_interval,
        agent_name,
    );

    //agent.load("checkpoints/dqn_v1.1.0_8J3.safetensors").expect("Failed to load model.check the path!");

    println!("========================================================");
    println!("Starting training for {} episodes",num_episodes);
    println!("Save_Interval:every {} episodes",save_interval);
    println!("Agent Name:{}",&trainer.agent_name);
    println!("Environment: DaifugoEnv with RuleConfig: eight_cut={}, eleven_back={}",rule.eight_cut,rule.eleven_back);
    println!("Input State Dimension: {}",INPUT_STATE_DIM);
    println!("=========================================================");

    trainer.train(&mut agent,&mut env,num_episodes).unwrap();

    let final_model_path = format!("{}/final_model.safetensors",trainer.save_dir);
    agent.save(&final_model_path).unwrap();
    println!("========================================================");
    println!("Training completed. Final model Savedto :{}",final_model_path);
    println!("========================================================");
}