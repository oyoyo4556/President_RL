use crate::lr_scheduler::CosineAnnealingWarmRestarts;
use crate::agent::{Agent,DQNAgent,Opponent,RandomAgent};
use crate::env::{DaifugoEnv,RawState};
use candle_core::Result;
use crate::common::{TRAIN_AGENT_ID};

pub struct Trainer {
    scheduler:CosineAnnealingWarmRestarts,
    batch_size:usize,
    tau:f32,
    pub save_dir:String,
    pub save_interval:usize,
    pub agent_name:String,
}

impl Trainer {
    pub fn new(eta_max:f64,eta_min:f64,t_0:usize,t_mult:usize,batch_size:usize,tau:f32,save_dir:String,save_interval:usize,agent_name:String) -> Self {
        Self { 
            scheduler: CosineAnnealingWarmRestarts::new(eta_max, eta_min, t_0, t_mult),
            batch_size,
            tau,
            save_dir,
            save_interval,
            agent_name,
        }
    }

    pub fn train_step(&mut self,agent:&mut DQNAgent) -> Result<f32> {
        self.scheduler.step();
        let current_lr = self.scheduler.get_lr();
        agent.set_learning_rate(current_lr);

        if self.scheduler.is_at_vally() {
            let cycle =self.scheduler.get_cycle_index();
            agent.save(&format!("{}/{}_cycle{}.safetensors",self.save_dir,self.agent_name,cycle))?;
            println!("Model saved at cycle {}",cycle);
        }

        let loss = agent.update(self.batch_size)?;
        agent.update_target_network(self.tau)?;
        Ok(loss)
    }

    pub fn run_episode(&mut self,agent:&mut DQNAgent,env:&mut DaifugoEnv) -> Result<(f32,f32)> {
        let mut state = env.reset();
        let mut total_loss = 0.0;
        let mut steps = 0;
        let mut done = false;
        let mut total_reward = 0.0;
        
        let mut daifugo_s:Option<RawState> = None;
        let mut daifugo_a:u16 = 0;

        let mut fugo_s:Option<RawState> = None;
        

        let mut previous_my_rank:Option<usize> = None;
        if !env.state.previous_rankings.is_empty() {
            previous_my_rank = env.state.previous_rankings.iter().position(|&p| p == env.agent_id as u8);
        }
        
        while !done {
            let action = agent.select_action(&state,TRAIN_AGENT_ID).map_err(candle_core::Error::msg)?;

            //手札交換は複雑。stateとactionとnext_stateが同じ視点から紐づくようにbufferに追加する必要がある。
            //env側ではlegal_action_maskを次に交換する人の手札を基準にexchange_stepでstateに含めて返しているため
            //大富豪のときと富豪の時でbufferへ追加するタイミングが変わる。
            if state.exchange_phase {
                let pre_my_rank = &previous_my_rank.unwrap();//ここでエラーが発動したらenvのバグ
                let current_ex_turn = env.exchange_turn_idx;
                match pre_my_rank { 
                    0 => {
                        if current_ex_turn == 0 {
                            let (next_exchange_state,exchange_reward,_) = env.exchange_step(action as usize);

                            agent.add_experience(
                                state,
                                action,
                                exchange_reward,
                                next_exchange_state.clone(),
                                false,
                            );

                            state = next_exchange_state;
                            done = false;
                            total_reward += exchange_reward;

                        } else if current_ex_turn == 1 {
                            daifugo_s = Some(state);
                            daifugo_a = action;
                            let (next_fugo_state,exchange_reward,_) = env.exchange_step(action as usize);

                            state = next_fugo_state;
                            done = false;
                            total_reward += exchange_reward;
                        } else {
                            let (next_state,_,_) = env.exchange_step(action as usize);

                            agent.add_experience(
                                daifugo_s.take().unwrap(),//unwrap()でエラーが出るとしたら、不正なデータの可能性があるので止める
                                daifugo_a,
                                0.0,
                                next_state.clone(),
                                false,
                            );

                            state = next_state;
                        }
                    }
                    1 => {
                        if current_ex_turn == 0 || current_ex_turn  == 1 {
                             let (next_exchange_state,_,_) = env.exchange_step(action as usize);
                            if current_ex_turn == 1 {
                                fugo_s = Some(next_exchange_state.clone());
                            }
                        } else {
                            
                            let (next_state,exchange_reward,_) = env.exchange_step(action as usize);

                            agent.add_experience(
                                fugo_s.take().unwrap(),//大富豪の時と同様
                                action,
                                0.0,
                                next_state.clone(),
                                false,
                            );

                            state = next_state;
                            total_reward += exchange_reward;
                        }
                    }
                    _ => {
                        let (next_state,_,_) = env.exchange_step(action as usize);
                        state = next_state;
                    } 
                
                }
                continue;
            }

            let (next_state,reward,is_done) = env.step(action);
            
            agent.add_experience(
                state.clone(),
                action,
                reward,
                next_state.clone(),
                is_done,
            );

            state = next_state;
            
            done = is_done;
            total_reward += reward;
        }

        if agent.buffer.len() >= self.batch_size {
                let loss = self.train_step(agent)?;
                total_loss += loss;
                steps += 1;
        }

        Ok((if steps >0 {total_loss/steps as f32} else {0.0},total_reward))
    }

    pub fn train(&mut self,agent:&mut DQNAgent,env:&mut DaifugoEnv,num_episodes:usize) -> Result<()> {

        let mut reward_history = Vec::new();
        let mut total_loss_sum = 0.0; 

        for episode in 1..=num_episodes {
            let (loss,episode_reward) = self.run_episode(agent,env)?;
            reward_history.push(episode_reward);
            total_loss_sum += loss;
            
            if episode % 100 == 0 {
                let avg_reward:f32 = reward_history.iter().rev().take(100).sum::<f32>()/100.0;
                let avg_loss:f32 = total_loss_sum/100.0;
                let current_lr = self.scheduler.get_lr();
                println!("Episode :{:>5},Ave_Reward:{:>7.2}, Ave_Loss:{:>8.4}, lr:{:>8.2e},Epsilon:{:>4.2}"
                ,episode,avg_reward,avg_loss,current_lr,agent.epsilon);
                total_loss_sum =0.0;
                if reward_history.len() > 1000 {
                    reward_history.drain(0..reward_history.len()-500);
                }

            }
            if episode %self.save_interval == 0 {
                let path = format!("{}/{}_ep{}.safetensors",self.save_dir,self.agent_name,episode);
                agent.save(&path)?;
                println!("Model saved on episode {}",episode);
            }

            //対戦相手の更新
            if episode % 3000 == 0 {
                if let Opponent::DQN(ref mut opp_agent) = env.opponent {
                    agent.copy_weights_to(opp_agent)?;
                    opp_agent.epsilon = 0.0;
                    println!("Opponent updated at episode {}",episode);
                } else {
                    let mut new_opp_agent = DQNAgent::new(100,1);
                    agent.copy_weights_to(&mut new_opp_agent)?;
                    new_opp_agent.epsilon = 0.0;
                    env.opponent = Opponent::DQN(new_opp_agent);
                    println!("Opponent switched to new agent");
                }
            }
        }
        
        Ok(())

    }

    

    pub fn vs_random(&mut self,agent:&mut DQNAgent,env:&mut DaifugoEnv,num_episodes:usize) -> Result<()> {
        let mut agent_ranks = Vec::new();
        let mut rank_counts = vec![0;4];
        env.opponent = Opponent::Random(RandomAgent::new());
        println!("=============================================================");
        println!("Starting evaluation vs RandomAgent for {} episodes",num_episodes);
        println!("=============================================================");
        for episode in 1..=num_episodes {
            let mut state = env.reset();
            let mut done = false;
            while !done {
                
                let action = agent.select_action(&state,TRAIN_AGENT_ID).map_err(candle_core::Error::msg)?;
                if state.exchange_phase {
                    let (next_exchange_state,_,_) = env.exchange_step(action as usize);
                    state = next_exchange_state;
                    continue;
                }

                let (next_state,_,is_done) = env.step(action);
                state = next_state;
                done = is_done;
            }
            
            let final_ranks = env.state.finished_order.clone();
            let agent_rank = final_ranks.iter().position(|&p| p == 0).expect("Failed to find agent's rank");
            agent_ranks.push(agent_rank);
            rank_counts[agent_rank] += 1;

            if episode % 1000 == 0 {
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
        println!("========================================================");
        Ok(())
    }
}