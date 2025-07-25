params = {
    "policy": "MlpPolicy",
    "learning_rate": 5e-4, #0.0005               
    "gamma": 0.99,
    "buffer_size": 20000,             
    "exploration_fraction": 0.2,        
    "exploration_final_eps": 0.01,     
    "target_update_interval": 250,      
    "train_freq": 1,                   
    "learning_starts": 500,            
    "batch_size": 64,                  
    "steps": 200000,
    "save_dir" : "./frozenlake/frozenlake_dqn/models",
    "check_freq" : 1000
}
