from cardchess_rl.agent.training import PPOTrainer

if __name__ == '__main__':
    trainer = PPOTrainer(
        games_per_iter=32,
        move_depth=1,
    )
    trainer.train(n_iterations=50, eval_interval=10)
