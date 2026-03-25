import random
import torch

# Import data utilities
from data_utils import Vocab, PrefixDataset, load_names, make_loaders
# Import evaluation utilities
from evaluation import TrainConfig, train_and_evaluate, print_result_block
# Import the three models
from model_vanilla_rnn import VanillaRNNNameModel
from model_blstm import BLSTMNameModel
from model_rnn_attention import RNNAttentionNameModel

# Path to the training data file
DATA_PATH = "TrainingNames.txt"
SEED = 42 # Random seed 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available, otherwise CPU

# Hyperparameters
EMBED_DIM = 96
HIDDEN_SIZE = 256
NUM_LAYERS = 2
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
EPOCHS = 40
DROPOUT = 0.1
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

random.seed(SEED)
torch.manual_seed(SEED)


def main():
    names = load_names(DATA_PATH)
    vocab = Vocab(names)
    dataset = PrefixDataset(names, vocab)
    # Create train, validation, and test dataloaders
    loaders = make_loaders(
        dataset,
        batch_size=BATCH_SIZE,
        pad_idx=vocab.pad_idx,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        seed=SEED,
    )

    print(f"Device: {DEVICE}")
    print(f"Loaded {len(names)} names | unique names: {len(set(names))} | vocab size: {len(vocab)} | examples: {len(dataset)}")

    # Create training configuration object
    cfg = TrainConfig(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT,
    )

    # Initialize models
    models = [
        VanillaRNNNameModel(len(vocab), EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE),
        BLSTMNameModel(len(vocab), EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE),
        RNNAttentionNameModel(len(vocab), EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE),
    ]

    # Train and evaluate each model
    results = []
    for model in models:
        result, _ = train_and_evaluate(model, loaders, cfg, vocab, names, DEVICE)
        results.append(result)
        print_result_block(result)

    print("\n" + "#" * 72)
    print("FINAL COMPARISON")
    print("#" * 72)
    header = f"{'Model':<22} {'Params':>12} {'TestLoss':>10} {'Novelty%':>10} {'Diversity':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['model']:<22} {r['parameters']:>12,} {r['test_loss']:>10.4f} {r['novelty_rate']:>10.2f} {r['diversity']:>10.4f}")

     # Qualitative discussion
    print("\nQualitative analysis summary template:")
    print("- Realism: inspect whether outputs resemble valid Indian full names.")
    print("- Failure modes: look for character repetition, hybridized surnames, spacing errors, and memorized training names.")
    print("- Trade-off: higher novelty can reduce realism; lower novelty can indicate memorization.")


#Main function
if __name__ == "__main__":
    main()
