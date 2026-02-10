"""
推理链 (Chain-of-Thought) 模块
================================
核心思想:
  不仅给出 "真/假" 的分类结果, 还生成一条可解释的推理链,
  让用户理解模型为什么做出这个判断.

推理链结构:
  Step 1: 文本特征分析 (情感词/煽动性/来源可信度/写作风格)
  Step 2: Attention关键词分析 (模型重点关注了哪些词)
  Step 3: 综合推理 (将上述线索整合为逻辑推理链)
  Conclusion: 最终判断 + 置信度

灵感来源:
  - L-Defense (WWW'2024): Explainable Fake News Detection with LLM
  - 本模块使用模型自身的Attention权重 + 规则特征分析, 无需外部LLM API
"""

import torch
import numpy as np


# ========================= 特征词典 =========================

# 煽动性/夸张用语 (假新闻常见)
SENSATIONAL_WORDS = {
    'shocking', 'unbelievable', 'breaking', 'exclusive', 'urgent',
    'bombshell', 'horrifying', 'terrifying', 'scandal', 'outrage',
    'devastating', 'explosive', 'stunning', 'alarming', 'incredible',
    'exposed', 'secret', 'leaked', 'conspiracy', 'hoax', 'cover',
    'destroyed', 'slammed', 'blasted', 'ripped', 'epic', 'insane'
}

# 可信度指标 (正规新闻常见)
CREDIBILITY_PHRASES = [
    'according to', 'research shows', 'study finds', 'officials said',
    'reuters', 'associated press', 'confirmed by', 'evidence suggests',
    'data shows', 'report says', 'peer reviewed', 'investigation found',
    'spokesperson said', 'published in', 'university of', 'department of',
    'official statement', 'press release', 'government report'
]

# 情感操控词汇
EMOTIONAL_WORDS = {
    'hate', 'love', 'angry', 'furious', 'amazing', 'terrible',
    'disgusting', 'wonderful', 'horrible', 'fantastic', 'awful',
    'outrageous', 'evil', 'hero', 'villain', 'miracle', 'disaster',
    'tragic', 'brilliant', 'stupid', 'genius', 'idiot', 'corrupt',
    'patriot', 'traitor', 'danger', 'threat', 'crisis', 'doom'
}

# 点击诱导词 (clickbait)
CLICKBAIT_PATTERNS = [
    'you won', 'believe', 'click here', 'share this', 'going viral',
    'mind blowing', 'what happened next', 'number', 'will shock',
    'doctors hate', 'one weird trick', 'exposed', 'they don want'
]


# ========================= 推理链分析器 =========================
class ChainOfThoughtAnalyzer:
    """
    推理链分析器: 结合模型Attention + 文本特征, 生成可解释的推理过程

    使用方式:
        analyzer = ChainOfThoughtAnalyzer(model, vocab, device)
        result = analyzer.analyze(cleaned_text, original_text)
        print(result['reasoning_chain'])
    """

    def __init__(self, model, vocab, device, max_length=500):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_length = max_length

    def analyze(self, text, original_text=None):
        """
        对单条文本生成完整的推理链分析

        参数:
            text: 清洗后的文本 (用于模型输入)
            original_text: 原始文本 (用于展示)

        返回:
            dict: {
                'text_preview': 文本预览,
                'prediction': 预测标签,
                'confidence': 置信度,
                'top_attention_words': Attention最高的词,
                'text_features': 文本特征分析结果,
                'reasoning_chain': 推理链文本
            }
        """
        display_text = original_text if original_text else text

        # ---- Step 1: 模型推理 + 获取Attention权重 ----
        self.model.eval()
        indices = self.vocab.encode(text, self.max_length)
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits, attention_weights = self.model(input_tensor, return_attention=True)

        prob = torch.sigmoid(logits).item()
        prediction = 'Real' if prob > 0.5 else 'Fake'
        confidence = prob if prob > 0.5 else 1 - prob

        # ---- Step 2: 提取Attention关键词 ----
        attention = attention_weights.squeeze().cpu().numpy()
        tokens = text.split()[:self.max_length]

        # 聚合同词的attention (取最大值)
        word_attention = {}
        for i, token in enumerate(tokens):
            if i < len(attention):
                if token not in word_attention:
                    word_attention[token] = 0
                word_attention[token] = max(word_attention[token], attention[i])

        top_words = sorted(word_attention.items(), key=lambda x: x[1], reverse=True)[:10]

        # ---- Step 3: 文本特征分析 ----
        features = self._analyze_features(text)

        # ---- Step 4: 生成推理链 ----
        reasoning = self._generate_reasoning(prediction, confidence, top_words, features)

        return {
            'text_preview': display_text[:300] + '...' if len(display_text) > 300 else display_text,
            'prediction': prediction,
            'confidence': f"{confidence:.2%}",
            'top_attention_words': [(w, f"{a:.4f}") for w, a in top_words[:5]],
            'text_features': features,
            'reasoning_chain': reasoning
        }

    def _analyze_features(self, text):
        """分析文本的多维特征"""
        words = text.split()
        text_lower = text.lower()

        # 1. 煽动性语言
        sensational_found = [w for w in words if w in SENSATIONAL_WORDS]
        sensational_ratio = len(sensational_found) / max(len(words), 1)

        # 2. 来源可信度
        credibility_found = [p for p in CREDIBILITY_PHRASES if p in text_lower]

        # 3. 情感操控
        emotional_found = [w for w in words if w in EMOTIONAL_WORDS]
        emotional_ratio = len(emotional_found) / max(len(words), 1)

        # 4. 点击诱导
        clickbait_found = [p for p in CLICKBAIT_PATTERNS if p in text_lower]

        # 5. 文本统计
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        def score(ratio, thresholds=(0.01, 0.005)):
            if ratio > thresholds[0]:
                return 'HIGH'
            elif ratio > thresholds[1]:
                return 'MEDIUM'
            return 'LOW'

        return {
            'sensational_score': score(sensational_ratio),
            'sensational_words': sensational_found[:5],
            'credibility_score': 'HIGH' if len(credibility_found) >= 2 else
                                 'MEDIUM' if len(credibility_found) == 1 else 'LOW',
            'credibility_indicators': credibility_found[:3],
            'emotional_score': score(emotional_ratio),
            'emotional_words': emotional_found[:5],
            'clickbait_score': 'HIGH' if len(clickbait_found) >= 2 else
                               'MEDIUM' if len(clickbait_found) == 1 else 'LOW',
            'clickbait_patterns': clickbait_found[:3],
            'text_length': len(words),
            'avg_word_length': f"{avg_word_len:.1f}"
        }

    def _generate_reasoning(self, prediction, confidence, top_words, features):
        """
        生成结构化的推理链文本

        推理链结构:
          Step 1: 文本特征分析
          Step 2: 模型Attention关键词
          Step 3: 综合推理
          Conclusion: 最终判断
        """
        lines = []

        # ==================== Step 1 ====================
        lines.append("=" * 50)
        lines.append("Step 1 - Text Feature Analysis:")
        lines.append(f"  [Sensational Language] {features['sensational_score']}")
        if features['sensational_words']:
            lines.append(f"    Found: {', '.join(features['sensational_words'])}")

        lines.append(f"  [Source Credibility]  {features['credibility_score']}")
        if features['credibility_indicators']:
            lines.append(f"    Found: {', '.join(features['credibility_indicators'])}")

        lines.append(f"  [Emotional Tone]     {features['emotional_score']}")
        if features['emotional_words']:
            lines.append(f"    Found: {', '.join(features['emotional_words'])}")

        lines.append(f"  [Clickbait Pattern]  {features['clickbait_score']}")
        if features['clickbait_patterns']:
            lines.append(f"    Found: {', '.join(features['clickbait_patterns'])}")

        lines.append(f"  [Text Length]        {features['text_length']} words")

        # ==================== Step 2 ====================
        lines.append("")
        lines.append("Step 2 - Model Attention Key Words:")
        for word, weight in top_words[:5]:
            bar = '█' * int(float(weight) * 500)  # 可视化注意力条
            lines.append(f"  '{word}' [{weight}] {bar}")

        # ==================== Step 3 ====================
        lines.append("")
        lines.append("Step 3 - Reasoning Chain:")
        reasons = []

        if prediction == 'Fake':
            if features['sensational_score'] in ('HIGH', 'MEDIUM'):
                reasons.append(
                    "The text contains sensational/exaggerated language, "
                    "which is a common pattern in fabricated news articles."
                )
            if features['credibility_score'] == 'LOW':
                reasons.append(
                    "The text lacks references to credible sources (e.g., "
                    "official agencies, research institutions), reducing its reliability."
                )
            if features['emotional_score'] in ('HIGH', 'MEDIUM'):
                reasons.append(
                    "Highly emotional language is detected, often used "
                    "to manipulate reader opinions rather than convey facts."
                )
            if features['clickbait_score'] in ('HIGH', 'MEDIUM'):
                reasons.append(
                    "Clickbait-style phrases are present, indicating "
                    "the text may prioritize engagement over accuracy."
                )
            if not reasons:
                reasons.append(
                    "The model's learned patterns from thousands of training samples "
                    "indicate this text matches characteristics commonly found in fake news."
                )
        else:  # Real
            if features['credibility_score'] in ('HIGH', 'MEDIUM'):
                reasons.append(
                    "The text references credible sources, "
                    "which is consistent with legitimate news reporting."
                )
            if features['sensational_score'] == 'LOW':
                reasons.append(
                    "The text uses neutral, factual language "
                    "consistent with professional journalism standards."
                )
            if features['emotional_score'] == 'LOW':
                reasons.append(
                    "The text maintains an objective tone without "
                    "excessive emotional manipulation."
                )
            if not reasons:
                reasons.append(
                    "The model's learned patterns from thousands of training samples "
                    "indicate this text matches characteristics commonly found in real news."
                )

        for i, reason in enumerate(reasons, 1):
            lines.append(f"  {i}. {reason}")

        # ==================== Conclusion ====================
        lines.append("")
        lines.append("=" * 50)
        lines.append(
            f"Conclusion: This article is classified as [{prediction}] "
            f"with {confidence:.2%} confidence."
        )
        lines.append("=" * 50)

        return '\n'.join(lines)


# ========================= 批量分析 =========================
def batch_analyze(analyzer, texts, n=5):
    """
    对多条文本进行推理链分析并打印结果

    参数:
        analyzer: ChainOfThoughtAnalyzer实例
        texts: 文本列表
        n: 分析条数
    """
    print("\n" + "=" * 60)
    print("  Chain-of-Thought Reasoning Demo")
    print("=" * 60)

    for i in range(min(n, len(texts))):
        print(f"\n{'─' * 60}")
        print(f"  Sample {i+1}:")
        print(f"{'─' * 60}")

        result = analyzer.analyze(texts[i])

        print(f"\nText Preview: {result['text_preview']}")
        print(f"\nPrediction: {result['prediction']} "
              f"(Confidence: {result['confidence']})")
        print(f"\n{result['reasoning_chain']}")

    print(f"\n{'═' * 60}")
