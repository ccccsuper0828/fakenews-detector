"""
Group 4 Runner: Knowledge Graph + RL experiments.

Usage:
    python -m experiments.group4_kg_rl.run_group4 [--quick] [--mode meta|dqn]
"""

import os
import sys
import argparse
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.config import ExperimentConfig, KGRLConfig
from experiments.metrics import compute_metrics, ResultLogger, plot_comparison_table
from experiments.group4_kg_rl.kg_builder import TripleExtractor, CommonsenseChecker
from experiments.group4_kg_rl.commonsense_checker import CommonsenseFeatureExtractor
from experiments.group4_kg_rl.rl_agent import (
    VerificationEnv, DQNAgent, MetaClassifier, train_meta_classifier,
)


def _train_proxy_model(cfg: ExperimentConfig):
    """Train TF-IDF + LogisticRegression as a fast proxy for G1/G2 model confidence."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    csv_path = os.path.join(base, cfg.data.csv_path)
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 50].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values, df["label"].values.astype(int),
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state, stratify=df["label"].values.astype(int),
    )

    print("  [ProxyModel] Training TF-IDF + LR on training split...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr_tfidf = tfidf.fit_transform(X_train)
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    lr.fit(X_tr_tfidf, y_train)

    train_acc = lr.score(X_tr_tfidf, y_train)
    test_acc = lr.score(tfidf.transform(X_test), y_test)
    print(f"  [ProxyModel] Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    return tfidf, lr


def load_test_data(cfg: ExperimentConfig, num_samples: int = 100):
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    csv_path = os.path.join(base, cfg.data.csv_path)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 50].reset_index(drop=True)

    texts = df["text"].values
    labels = df["label"].values.astype(int)

    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=cfg.data.test_size,
        random_state=cfg.data.random_state, stratify=labels,
    )

    n = min(num_samples, len(X_test))
    idx = np.random.RandomState(42).choice(len(X_test), n, replace=False)
    return X_test[idx], y_test[idx]


def _text_features(text: str) -> dict:
    """Lightweight content features to replace random model_conf."""
    words = text.lower().split()
    n = max(len(words), 1)
    sensational = ["breaking", "shocking", "bombshell", "exclusive", "scandal",
                   "leaked", "exposed", "cover-up", "conspiracy", "hoax",
                   "urgent", "alert", "you won't believe", "must see"]
    n_sensational = sum(1 for s in sensational if s in text.lower())
    n_exclaim = text.count("!")
    n_caps = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 3]
    avg_sent_len = np.mean([len(s.split()) for s in sents]) if sents else 10.0
    return {
        "sensational_ratio": min(1.0, n_sensational * 0.15),
        "exclaim_ratio": min(1.0, n_exclaim / max(n / 100, 1)),
        "caps_ratio": n_caps / n,
        "avg_sent_len": min(avg_sent_len / 30.0, 1.0),
        "text_len": min(n / 500.0, 1.0),
    }


def run_meta_classifier_experiment(texts, labels, cfg, proxy_model=None):
    """Run the meta-classifier with real text + KG features.
    If proxy_model=(tfidf, lr) is given, use G1/G2 proxy scores instead of text features."""
    print("\n  Extracting KG + text features...")
    feature_extractor = CommonsenseFeatureExtractor(
        kg_sources=["conceptnet"],
        use_llm=False,
    )

    use_proxy = proxy_model is not None
    if use_proxy:
        tfidf, lr = proxy_model
        proxy_probs = lr.predict_proba(tfidf.transform(texts))[:, 1]
        print(f"  [Fusion] G1/G2 proxy scores: mean={proxy_probs.mean():.3f}, std={proxy_probs.std():.3f}")

    all_features = []
    for i, text in enumerate(texts):
        kg = feature_extractor.extract_features(str(text))
        tf = _text_features(str(text))

        if use_proxy:
            p = float(proxy_probs[i])
            feature_vec = [
                p,                            # G1/G2 proxy confidence (replaces random)
                1.0 if p > 0.5 else 0.0,      # proxy prediction
                tf["sensational_ratio"],
                tf["caps_ratio"],
                tf["avg_sent_len"],
                kg["violation_score"],
                kg["verified_ratio"],
                kg["avg_kg_score"],
                kg["num_triples"] / 10.0,
            ]
        else:
            feature_vec = [
                tf["sensational_ratio"],
                tf["exclaim_ratio"],
                tf["caps_ratio"],
                tf["avg_sent_len"],
                tf["text_len"],
                kg["violation_score"],
                kg["verified_ratio"],
                kg["avg_kg_score"],
                kg["num_triples"] / 10.0,
            ]
        all_features.append(feature_vec)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(texts)}] features extracted")

    features = np.array(all_features, dtype=np.float32)

    split = int(len(features) * 0.7)
    train_features, test_features = features[:split], features[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    tag = "G4_meta_fusion" if use_proxy else "G4_meta_classifier"
    print(f"\n  Training {tag} (train={len(train_labels)}, test={len(test_labels)})...")
    model = train_meta_classifier(train_features, train_labels, num_epochs=300)

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(test_features))
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)

    result = compute_metrics(
        test_labels, preds, probs,
        experiment_name=tag,
        group="group4_kg_rl",
    )
    return result


def run_dqn_experiment(texts, labels, cfg):
    """Run DQN routing with real text-derived features."""
    print("\n  Pre-computing features for DQN...")
    feature_extractor = CommonsenseFeatureExtractor(
        kg_sources=["conceptnet"], use_llm=False,
    )

    text_feats = []
    kg_feats = []
    for i, text in enumerate(texts):
        tf = _text_features(str(text))
        kf = feature_extractor.extract_features(str(text))
        content_score = 1.0 - tf["sensational_ratio"] * 0.5 - tf["caps_ratio"] * 0.3
        text_feats.append(max(0.0, min(1.0, content_score)))
        kg_feats.append(kf)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(texts)}] features ready")

    print("\n  Training DQN agent...")
    kgrl = cfg.kg_rl
    env = VerificationEnv(step_cost=kgrl.step_cost, max_steps=kgrl.max_steps)
    agent = DQNAgent(
        state_dim=env.state_dim,
        num_actions=env.NUM_ACTIONS,
        hidden_dim=kgrl.rl_hidden_dim,
        lr=kgrl.rl_lr,
        gamma=kgrl.rl_gamma,
    )

    num_episodes = min(len(texts) * 3, 600)
    total_rewards = []

    for ep in range(num_episodes):
        idx = ep % len(texts)
        model_conf = text_feats[idx]
        true_label = int(labels[idx])
        kf = kg_feats[idx]

        kg_feat_dict = {"avg_kg_score": kf["avg_kg_score"]}
        web_feat_dict = {
            "entailment_score": kf["verified_ratio"],
            "nli_score": 1.0 - kf["violation_score"],
        }

        state = env.reset(model_conf, true_label, kg_feat_dict, web_feat_dict)
        episode_reward = 0

        while not env.done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

        if (ep + 1) % 50 == 0:
            agent.update_target()
            avg_r = np.mean(total_rewards[-50:])
            print(f"  Episode {ep+1}/{num_episodes} AvgReward={avg_r:.3f} "
                  f"Epsilon={agent.epsilon:.3f}")

    agent.epsilon = 0.0
    predictions = []
    confidences = []
    for i in range(len(labels)):
        model_conf = text_feats[i]
        kf = kg_feats[i]
        state = env.reset(
            model_conf, int(labels[i]),
            {"avg_kg_score": kf["avg_kg_score"]},
            {"entailment_score": kf["verified_ratio"],
             "nli_score": 1.0 - kf["violation_score"]},
        )
        while not env.done:
            action = agent.select_action(state)
            state, _, done = env.step(action)

        if action == env.ACTION_PREDICT_REAL:
            pred = 1
        elif action == env.ACTION_PREDICT_FAKE:
            pred = 0
        else:
            pred = 1 if state[0] > 0.5 else 0
        predictions.append(pred)
        confidences.append(abs(state[0] - 0.5) * 2)

    predictions = np.array(predictions)
    result = compute_metrics(
        labels, predictions, np.array(confidences),
        experiment_name="G4_dqn_routing",
        group="group4_kg_rl",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mode", choices=["meta", "dqn", "both", "fusion"], default="both")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("  Group 4: Knowledge Graph + RL Experiments")
    print("=" * 60)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "outputs", "group4_kg_rl",
    )
    os.makedirs(output_dir, exist_ok=True)
    logger = ResultLogger(os.path.join(output_dir, "results.csv"))

    num_samples = 30 if args.quick else args.num_samples
    cfg = ExperimentConfig(
        name="G4", group="group4_kg_rl",
        kg_rl=KGRLConfig(),
    )

    print("  Loading test data...")
    texts, labels = load_test_data(cfg, num_samples=num_samples)
    print(f"  Using {len(texts)} samples")

    all_results = []

    if args.mode == "fusion":
        print("\n  === G1/G2 Fusion Mode ===")
        proxy = _train_proxy_model(cfg)
        result = run_meta_classifier_experiment(texts, labels, cfg, proxy_model=proxy)
        all_results.append(result)
        logger.log(result)

    if args.mode in ("meta", "both"):
        result = run_meta_classifier_experiment(texts, labels, cfg)
        all_results.append(result)
        logger.log(result)

    if args.mode in ("dqn", "both"):
        result = run_dqn_experiment(texts, labels, cfg)
        all_results.append(result)
        logger.log(result)

    if all_results:
        plot_comparison_table(
            all_results,
            save_path=os.path.join(output_dir, "group4_comparison.png"),
            title="Group 4: KG + RL Comparison",
        )

    print(f"\n  Group 4 results saved to {output_dir}")


if __name__ == "__main__":
    main()
