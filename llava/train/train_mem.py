from llava.train.train import train
import os

if __name__ == "__main__":
    print("starting train_mem")
    for k, v in os.environ.items():
        k = k.upper()
        if (
            "RANK" in k
            or "GPU" in k
            or "TASK" in k
            or "WORLD" in k
            or "NODE" in k
            or "SLURM" in k
            or "MASTER" in k
        ) and not ("VSCODE" in k):
            print(f"{k}: {v}")
    train(attn_implementation="flash_attention_2")
