import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from fruit_gym import envs
import numpy as np
import time
import os
from wrappers import VideoRecorder, RotateImage, SERLObsWrapper

def benchmark_env():
    """
    Benchmark script that runs 20 episodes and measures timing performance.
    """
    # Configuration
    num_episodes = 5
    camera_res = 480
    cameras = ['wrist1', 'wrist2']
    proprio_keys = None
    max_episode_steps = 100
    
    # Disable video recording for benchmarking
    record = False
    
    print(f"Starting benchmark: {num_episodes} episodes with max {max_episode_steps} steps each")
    print("=" * 60)
    
    # Initialize environment
    env = gym.make(
        "PickMultiStrawbEnv", 
        randomize_domain=True, 
        cameras=cameras, 
        include_privileged_obs=False, 
        reward_type="dense", 
        ee_dof=4, 
        width=camera_res, 
        height=camera_res, 
        gripper_pause=True
    )
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = RotateImage(env, pixel_key="wrist1")
    
    # Timing variables
    total_start_time = time.time()
    episode_times = []
    step_times = []
    reset_times = []
    total_steps = 0
    total_rewards = []
    
    # Random action parameters (for consistent benchmarking)
    np.random.seed(42)  # For reproducible results
    
    try:
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            # Time the reset
            reset_start = time.time()
            obs, info = env.reset()
            reset_time = time.time() - reset_start
            reset_times.append(reset_time)
            
            terminated = False
            truncated = False
            episode_reward = 0
            episode_steps = 0
            i = 0
            
            print(f"Episode {episode + 1}/{num_episodes} - Reset time: {reset_time:.4f}s")
            
            while not (terminated or truncated):
                step_start = time.time()
                
                # Generate random action for benchmarking
                # Format: [forward/back, left/right, up/down, rotation, gripper]
                if i < 40:
                    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                    if i % 20 == 0:
                        current_y = np.random.choice([1.0, -1.0])
                    action[1] = current_y
                    action[-1] = np.random.choice([-1.0, 0.0, 1.0])
                i += 1

                # Execute step
                obs, reward, terminated, truncated, info = env.step(action)
                # cv2.imshow("wrist1", obs['wrist1'])
                # cv2.waitKey(1)  # Allow OpenCV to update the window
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
            
            episode_time = time.time() - episode_start_time
            episode_times.append(episode_time)
            total_rewards.append(episode_reward)
            
            print(f"  Completed in {episode_time:.2f}s | Steps: {episode_steps} | Reward: {episode_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nError during benchmark: {e}")
    finally:
        env.close()
    
    # Calculate and display results
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    if episode_times:
        print(f"Total episodes completed: {len(episode_times)}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Total steps executed: {total_steps}")
        print()
        
        print("EPISODE TIMING:")
        print(f"  Average episode time: {np.mean(episode_times):.4f} ± {np.std(episode_times):.4f} seconds")
        print(f"  Min episode time: {np.min(episode_times):.4f} seconds")
        print(f"  Max episode time: {np.max(episode_times):.4f} seconds")
        print()
        
        print("STEP TIMING:")
        print(f"  Average step time: {np.mean(step_times):.6f} ± {np.std(step_times):.6f} seconds")
        print(f"  Steps per second: {1.0 / np.mean(step_times):.2f}")
        print(f"  Min step time: {np.min(step_times):.6f} seconds")
        print(f"  Max step time: {np.max(step_times):.6f} seconds")
        print()
        
        print("RESET TIMING:")
        print(f"  Average reset time: {np.mean(reset_times):.4f} ± {np.std(reset_times):.4f} seconds")
        print(f"  Min reset time: {np.min(reset_times):.4f} seconds")
        print(f"  Max reset time: {np.max(reset_times):.4f} seconds")
        print()
        
        print("PERFORMANCE METRICS:")
        print(f"  Average reward per episode: {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
        print(f"  Average steps per episode: {total_steps / len(episode_times):.1f}")
        print(f"  Episodes per minute: {len(episode_times) / (total_time / 60):.2f}")
        
        # Performance percentiles
        print(f"\nSTEP TIME PERCENTILES:")
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            print(f"  {p}th percentile: {np.percentile(step_times, p):.6f} seconds")
    
    print("=" * 60)
    return {
        'total_time': total_time,
        'episode_times': episode_times,
        'step_times': step_times,
        'reset_times': reset_times,
        'total_steps': total_steps,
        'total_rewards': total_rewards
    }

if __name__ == "__main__":
    print("Environment Benchmarking Script")
    print("This will run 20 episodes with random actions for performance measurement")
    print()
    
    # Run benchmark
    results = benchmark_env()
    
    # Save results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write(f"Benchmark Results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total episodes: {len(results['episode_times'])}\n")
        f.write(f"Total time: {results['total_time']:.2f} seconds\n")
        f.write(f"Average episode time: {np.mean(results['episode_times']):.4f} seconds\n")
        f.write(f"Average step time: {np.mean(results['step_times']):.6f} seconds\n")
        f.write(f"Steps per second: {1.0 / np.mean(results['step_times']):.2f}\n")
        f.write(f"Total steps: {results['total_steps']}\n")
        f.write(f"Average reward: {np.mean(results['total_rewards']):.4f}\n")
    
    print(f"\nResults saved to: {results_file}")
