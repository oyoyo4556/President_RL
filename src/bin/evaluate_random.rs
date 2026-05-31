use std::fs;
use std::path::Path;
use president::env::{DaifugoEnv};
use president::agent::{DQNAgent,RandomAgent,Opponent};
use president::trainer::Trainer;

fn main(){
    let save_dir ="checkpoints".to_string();
    if !Path::new(&save_dir).exists() {
        fs::create_dir_all(&save_dir).expect("Failed to create save directory.");
        println!("Created directory: {}",save_dir);
    }

    let eta_max = 1e-4;
    let eta_min = 1e-5;
    let t_0 = 10000;
    let t_mult = 2;

    let batch_size = 64;
    let tau = 1.0;
    let save_interval = 3000;
    let num_episodes = 10000;
    let agent_name = "dqn_v1.0.2".to_string();

    let opponent = Opponent::Random(RandomAgent::new());
    let mut env = DaifugoEnv::new(0,opponent);
    let mut agent = DQNAgent::new(100_000,3);
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

    agent.load("checkpoints/dqn_v1.0.2_daifugo_ep100000.safetensors").expect("Failed to load model.check the path!");
    agent.epsilon = 0.0;

    trainer.vs_random(&mut agent,&mut env,num_episodes).unwrap();
}